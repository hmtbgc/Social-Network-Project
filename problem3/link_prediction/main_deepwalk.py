import os
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import logging
import argparse
import os
import configs as configs
from torch.utils.tensorboard import SummaryWriter
import time
from dgl.nn import DeepWalk
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")



def init_logging(log_path):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

def hyperparam_concat(args):
    if (args.model in ["deepwalk"]):
        return f"{args.model}_emb{args.emb}_walklength{args.length}_window{args.window}_negsize{args.neg_size}_edge_type_{args.edge_combine}_lr{args.lr}_first_epoch{args.first_epoch}_second_epoch{args.second_epoch}"
    if (args.model in ["node2vec"]):
        return f"{args.model}_emb{args.emb}_walklength{args.length}_window{args.window}_negsize{args.neg_size}_p{args.p}_q{args.q}_edge_type_{args.edge_combine}_lr{args.lr}_first_epoch{args.first_epoch}_second_epoch{args.second_epoch}"
        
    

def check_exist_and_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        

def read_data(path):
    u, v = [], []
    with open(path, "r") as f:
        for line in f:
            x = line.strip().split()
            a, b = int(x[0]), int(x[1])
            u.append(a)
            v.append(b)

    return u, v

def read_eids(path):
    e = []
    with open(path, "r") as f:
        for line in f:
            x = line.strip().split()
            e.append(int(x[0]))
    return e


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)

class DotPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)
    
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            return self.layer(g.edata["score"]).squeeze(1)
        
class MLPredictor(nn.Module):
    def __init__(self, h_feats, edge_combine_type):
        super().__init__()
        if (edge_combine_type in ["average", "hadmard"]):
            self.layer = nn.Linear(h_feats, 1)
        elif (edge_combine_type == "concat"):
            self.layer = nn.Sequential(
                    nn.Linear(h_feats * 2, h_feats),
                    nn.ReLU(inplace=True),
                    nn.Linear(h_feats, 1)
                ) 
        self.edge_combine_type = edge_combine_type
        
    def apply_edges(self, edges):
        if (self.edge_combine_type == "average"):
            h = (edges.src["h"] + edges.dst["h"]) * 0.5
        elif (self.edge_combine_type == "hadmard"):
            h = edges.src["h"] * edges.dst["h"]
        elif (self.edge_combine_type == "concat"):
            h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": self.layer(h).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, help="model", default="deepwalk")
    parser.add_argument("-lr", type=float, help="learning rate", default=0.01)
    parser.add_argument("-emb", type=int, help="embedding dim", default=128)
    parser.add_argument("-length", type=int, help="length of each random walk", default=40)
    parser.add_argument("-window", type=int, help="window size", default=5)
    parser.add_argument("-neg_size", type=int, help="number of negative samples for each positive sample", default=5)
    parser.add_argument("-p", type=float, help="likelihood of immediately revisiting a node in the walk(only for node2vec)", default=1)
    parser.add_argument("-q", type=float, help="control parameter to interpolate between breadth-first strategy and depth-first strategy(only for node2vec)", default=1)
    parser.add_argument("-first_epoch", type=int, help="first phase epoch", default=20)
    parser.add_argument("-second_epoch", type=int, help="second phase epoch", default=200)
    parser.add_argument("-eval", type=int, help="eval frequency", default=2)
    parser.add_argument("-edge_combine", type=str, help="edge combination type", default="dot")
    args = parser.parse_args()
    
    model_pt_root = configs.model_pt_saved_root
    log_root = configs.log_root
    tensorboard_log_root = configs.tensorboard_log_root
    check_exist_and_mkdir(model_pt_root)
    check_exist_and_mkdir(log_root)
    check_exist_and_mkdir(tensorboard_log_root)
    
    model_pt_root_with_model = os.path.join(model_pt_root, args.model)
    log_root_with_model = os.path.join(log_root, args.model)
    tensorboard_log_root_with_model = os.path.join(tensorboard_log_root, args.model)
    
    check_exist_and_mkdir(model_pt_root_with_model)
    check_exist_and_mkdir(log_root_with_model)
    check_exist_and_mkdir(tensorboard_log_root)

    model_pt_path = os.path.join(model_pt_root_with_model, hyperparam_concat(args) + ".pt")
    mlp_model_pt_path = os.path.join(model_pt_root_with_model, hyperparam_concat(args) + "_mlp.pt")
    logger = init_logging(os.path.join(log_root_with_model, hyperparam_concat(args) + ".log"))
    writer = SummaryWriter(os.path.join(tensorboard_log_root_with_model, hyperparam_concat(args)))
    
    train_pos_u, train_pos_v = read_data("./data/train_pos.txt")
    val_pos_u, val_pos_v = read_data("./data/valid_pos.txt")
    test_pos_u, test_pos_v = read_data("./data/test_pos.txt")

    train_neg_u, train_neg_v = read_data("./data/train_neg.txt")
    val_neg_u, val_neg_v = read_data("./data/valid_neg.txt")
    test_neg_u, test_neg_v = read_data("./data/test_neg.txt")
    
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    
    
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes())
    val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=g.num_nodes())
    val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=g.num_nodes())
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.num_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.num_nodes())
    
    val_test_eids = read_eids("./data/val_test_eids.txt")
    train_g = dgl.remove_edges(g, torch.tensor(val_test_eids))
    train_g = dgl.add_self_loop(train_g)
    
    model = DeepWalk(train_g, emb_dim=args.emb, walk_length=args.length, window_size=args.window, negative_size=args.neg_size)
    
    
    
    if (args.model == "deepwalk"):
        dataloader = DataLoader(torch.arange(train_g.num_nodes()), batch_size=128,
                            shuffle=True, collate_fn=model.sample)
    elif (args.model == "node2vec"):
        dataloader = DataLoader(torch.arange(train_g.num_nodes()), batch_size=128,
                    shuffle=True, collate_fn=lambda indices : dgl.sampling.node2vec_random_walk(train_g, nodes=indices, p=args.p, q=args.q, walk_length=args.length-1))
    else:
        raise ValueError("model must be deepwalk or node2vec!")
        
    
    first_phase_optimizer = SparseAdam(model.parameters(), lr=0.01)
    
    
    logger.info("training......")
    print("training......")
    begin_train_time = time.time()
    
    for e in range(args.first_epoch):
        epoch_loss = 0.0
        for batch_walk in dataloader:
            loss = model(batch_walk)
            epoch_loss += loss.item()
            first_phase_optimizer.zero_grad()
            loss.backward()
            first_phase_optimizer.step()
        writer.add_scalar("first phase train loss", epoch_loss / len(dataloader), e)
        logger.info(f"epoch {e}, first phase train loss: {epoch_loss / len(dataloader):.4f}")
    
    
    X = model.node_embed.weight.detach()
    
    X.requires_grad = False
    torch.save(X, model_pt_path)
    
    predictor = DotPredictor()
    test_predictor = DotPredictor()
    
    if (args.edge_combine in ["average", "hadmard", "concat"]):
        predictor = MLPredictor(args.emb, args.edge_combine)
        test_predictor = MLPredictor(args.emb, args.edge_combine)
    
    second_phase_optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)
    best_val_auc = 0
    
    for e in range(args.second_epoch):
        pos_score = predictor(train_pos_g, X)
        neg_score = predictor(train_neg_g, X)
        loss = compute_loss(pos_score, neg_score)
        writer.add_scalar("second phase train loss", loss, e)

        logger.info(f"epoch {e}, second phase train loss: {loss:.4f}")
        
        second_phase_optimizer.zero_grad()
        loss.backward()
        second_phase_optimizer.step()
        
        if (e % args.eval == 0):
            pos_score = predictor(val_pos_g, X)
            neg_score = predictor(val_neg_g, X)
            val_loss = compute_loss(pos_score, neg_score)
            val_auc = compute_auc(pos_score, neg_score)
            writer.add_scalar("valid loss", val_loss, e)
            writer.add_scalar("valid auc", val_auc, e)
            logger.info("evaluating......")
            logger.info(f"epoch {e}, valid loss: {val_loss:.4f}, valid auc: {val_auc * 100:.2f}%")
            if (best_val_auc < val_auc):
                best_val_auc = val_auc
                torch.save(predictor.state_dict(), mlp_model_pt_path)
                    
    end_train_time = time.time()
    logger.info(f"total train time used: {end_train_time - begin_train_time:.4f}s")
    
    
        
    logger.info("testing......")
    print("testing......")
    test_X = torch.load(model_pt_path)
    test_predictor.load_state_dict(torch.load(mlp_model_pt_path))
    
    with torch.no_grad():
        pos_score = test_predictor(test_pos_g, test_X)
        neg_score = test_predictor(test_neg_g, test_X)
        test_auc = compute_auc(pos_score, neg_score)
        logger.info(f"test auc: {test_auc * 100:.2f}%")
    
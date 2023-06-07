import os
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ["DGLBACKEND"] = "pytorch"
import dgl.function as fn
import logging
import argparse
import os
import configs as configs
from torch.utils.tensorboard import SummaryWriter
from models import *
import time
from sklearn.metrics import roc_auc_score
import numpy as np
import scipy.sparse as sp
import itertools

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
    if (args.model in ["gcn", "graphsage", "mlp", "gin"]):
        return f"{args.model}_layer{args.nlayer}_hid{args.hid}_out{args.out}_edge_type_{args.edge_combine}_lr{args.lr}_dropout{args.dropout}_epoch{args.epoch}"
    if (args.model in ["gat"]):
        return f"{args.model}_layer{args.nlayer}_hid{args.hid}_out{args.out}_edge_type_{args.edge_combine}_head{args.head}_lr{args.lr}_dropout{args.dropout}_epoch{args.epoch}"
    if (args.model in ["sgc"]):
        return f"{args.model}_k{args.k}_out{args.out}_edge_type_{args.edge_combine}_lr{args.lr}_epoch{args.epoch}"
    

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
    ).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            return g.edata["score"][:, 0]
        
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
    parser.add_argument("-model", type=str, help="model", default="gcn")
    parser.add_argument("-device", type=str, help="train/test device", default="cuda:0")
    parser.add_argument("-lr", type=float, help="learning rate", default=0.01)
    parser.add_argument("-hid", type=int, help="hidden feature dimension", default=16)
    parser.add_argument("-out", type=int, help="output feature dimension", default=16)
    parser.add_argument("-edge_combine", type=str, help="edge combination type", default="dot")
    parser.add_argument("-nlayer", type=int, help="number of layer", default=2)
    parser.add_argument("-dropout", type=float, help="dropout", default=0.0)
    parser.add_argument("-eval", type=int, help="eval frequency", default=2)
    parser.add_argument("-epoch", type=int, help="total epoch", default=100)
    parser.add_argument("-head", type=int, help="head number(only GAT)", default=3)
    parser.add_argument("-k", type=int, help="hop number(only SGC)", default=3)
    
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
    device = torch.device(args.device)
    
    
    train_pos_u, train_pos_v = read_data("./data/train_pos.txt")
    val_pos_u, val_pos_v = read_data("./data/valid_pos.txt")
    test_pos_u, test_pos_v = read_data("./data/test_pos.txt")

    train_neg_u, train_neg_v = read_data("./data/train_neg.txt")
    val_neg_u, val_neg_v = read_data("./data/valid_neg.txt")
    test_neg_u, test_neg_v = read_data("./data/test_neg.txt")
    
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    
    
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes()).to(device)
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes()).to(device)
    val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=g.num_nodes()).to(device)
    val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=g.num_nodes()).to(device)
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.num_nodes()).to(device)
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.num_nodes()).to(device)
    
    val_test_eids = read_eids("./data/val_test_eids.txt")
    train_g = dgl.remove_edges(g, torch.tensor(val_test_eids)).to(device)
    train_g = dgl.add_self_loop(train_g)

    
    if (args.model == "gcn"):
        model = GCN(g.ndata["feat"].shape[1], args.hid, args.nlayer, args.out, args.dropout)
        test_model = GCN(g.ndata["feat"].shape[1], args.hid, args.nlayer, args.out, args.dropout)
    elif (args.model == "graphsage"):
        model = GraphSage(g.ndata["feat"].shape[1], args.hid, args.nlayer, args.out, args.dropout)
        test_model = GraphSage(g.ndata["feat"].shape[1], args.hid, args.nlayer, args.out, args.dropout)
    elif (args.model == "mlp"):
        model = MLP(g.ndata["feat"].shape[1], args.hid, args.nlayer, args.out, args.dropout)
        test_model = MLP(g.ndata["feat"].shape[1], args.hid, args.nlayer, args.out, args.dropout)
    elif (args.model == "gat"):
        model = GAT(g.ndata["feat"].shape[1], args.hid, args.nlayer, args.head, args.out, args.dropout)
        test_model = GAT(g.ndata["feat"].shape[1], args.hid, args.nlayer, args.head, args.out, args.dropout)
    elif (args.model == "gin"):
        model = GIN(g.ndata["feat"].shape[1], args.hid, args.nlayer, args.out, args.dropout)
        test_model = GIN(g.ndata["feat"].shape[1], args.hid, args.nlayer, args.out, args.dropout)
    elif (args.model == "sgc"):
        model = SGC(g.ndata["feat"].shape[1], args.out, args.k)
        test_model = SGC(g.ndata["feat"].shape[1], args.out, args.k)
        
    model = model.to(device)
    test_model = test_model.to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    best_val_auc = 0
    
    predictor = DotPredictor().to(device)
    test_predictor = DotPredictor().to(device)
    
    if (args.edge_combine in ["average", "hadmard", "concat"]):
        predictor = MLPredictor(args.out, args.edge_combine).to(device)
        test_predictor = MLPredictor(args.out, args.edge_combine).to(device)
        optimizer = torch.optim.Adam(itertools.chain(model.parameters(), predictor.parameters()), lr=args.lr)

    
    logger.info("training......")
    print("training......")
    
    begin_train_time = time.time()
    
    for e in range(args.epoch):
        h = model(train_g, train_g.ndata["feat"])
        pos_score = predictor(train_pos_g, h)
        neg_score = predictor(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)
    
        writer.add_scalar("train loss", loss, e)

        logger.info(f"epoch {e}, train loss: {loss:.4f}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (e % args.eval == 0):
            pos_score = predictor(val_pos_g, h)
            neg_score = predictor(val_neg_g, h)
            val_loss = compute_loss(pos_score, neg_score)
            val_auc = compute_auc(pos_score, neg_score)
            writer.add_scalar("valid loss", val_loss, e)
            writer.add_scalar("valid auc", val_auc, e)
            logger.info("evaluating......")
            logger.info(f"epoch {e}, valid loss: {val_loss:.4f}, valid auc: {val_auc * 100:.2f}%")
            if (best_val_auc < val_auc):
                best_val_auc = val_auc
                torch.save(model.state_dict(), model_pt_path)
                if (args.edge_combine in ["average", "hadmard", "concat"]):
                    torch.save(predictor.state_dict(), mlp_model_pt_path)
    
    end_train_time = time.time()
    logger.info(f"total train time used: {end_train_time - begin_train_time:.4f}s")
    
    logger.info("testing......")
    print("testing......")
    test_model.load_state_dict(torch.load(model_pt_path))
    if (args.edge_combine in ["average", "hadmard", "concat"]):
        test_predictor.load_state_dict(torch.load(mlp_model_pt_path))
    with torch.no_grad():
        h = test_model(train_g, train_g.ndata["feat"])
        pos_score = test_predictor(test_pos_g, h)
        neg_score = test_predictor(test_neg_g, h)
        test_auc = compute_auc(pos_score, neg_score)
        logger.info(f"test auc: {test_auc * 100:.2f}%")
        
        

        
    
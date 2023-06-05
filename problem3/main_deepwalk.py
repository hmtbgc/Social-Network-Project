import os
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import argparse
import os
import configs
from torch.utils.tensorboard import SummaryWriter
from models import *
import time
from dgl.nn import DeepWalk
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression


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
        return f"{args.model}_layer{args.nlayer}_hid{args.hid}_lr{args.lr}_dropout{args.dropout}_epoch{args.epoch}"
    if (args.model in ["gat"]):
        return f"{args.model}_layer{args.nlayer}_hid{args.hid}_head{args.head}_lr{args.lr}_dropout{args.dropout}_epoch{args.epoch}"
    if (args.model in ["sgc"]):
        return f"{args.model}_k{args.k}_lr{args.lr}_epoch{args.epoch}"
    if (args.model in ["deepwalk"]):
        return f"{args.model}_emb{args.emb}_walklength{args.length}_window{args.window}_negsize{args.neg_size}_lr{args.lr}_epoch{args.epoch}"
    if (args.model in ["node2vec"]):
        return f"{args.model}_emb{args.emb}_walklength{args.length}_window{args.window}_negsize{args.neg_size}_p{args.p}_q{args.q}_lr{args.lr}_epoch{args.epoch}"
        
    

def check_exist_and_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, help="model", default="gcn")
    parser.add_argument("-lr", type=float, help="learning rate", default=0.01)
    parser.add_argument("-emb", type=int, help="embedding dim", default=128)
    parser.add_argument("-length", type=int, help="length of each random walk", default=40)
    parser.add_argument("-window", type=int, help="window size", default=5)
    parser.add_argument("-neg_size", type=int, help="number of negative samples for each positive sample", default=5)
    parser.add_argument("-p", type=float, help="likelihood of immediately revisiting a node in the walk(only for node2vec)", default=1)
    parser.add_argument("-q", type=float, help="control parameter to interpolate between breadth-first strategy and depth-first strategy(only for node2vec)", default=1)
    parser.add_argument("-epoch", type=int, help="total epoch", default=100)
    args = parser.parse_args()
    
    log_root = configs.log_root
    tensorboard_log_root = configs.tensorboard_log_root
    check_exist_and_mkdir(log_root)
    check_exist_and_mkdir(tensorboard_log_root)
    
    log_root_with_model = os.path.join(log_root, args.model)
    tensorboard_log_root_with_model = os.path.join(tensorboard_log_root, args.model)
    
    check_exist_and_mkdir(log_root_with_model)
    check_exist_and_mkdir(tensorboard_log_root)

    logger = init_logging(os.path.join(log_root_with_model, hyperparam_concat(args) + ".log"))
    writer = SummaryWriter(os.path.join(tensorboard_log_root_with_model, hyperparam_concat(args)))
    
    dataset = dgl.data.CoraGraphDataset()
    logger.info(f"Number of categories:{dataset.num_classes}")
    g = dataset[0]
    logger.info(f"graph node data: {g.ndata}")
    logger.info(f"graph edge data: {g.edata}")
    
    model = DeepWalk(g, emb_dim=args.emb, walk_length=args.length, window_size=args.window, negative_size=args.neg_size)
    
    
    
    if (args.model == "deepwalk"):
        dataloader = DataLoader(torch.arange(g.num_nodes()), batch_size=128,
                            shuffle=True, collate_fn=model.sample)
    elif (args.model == "node2vec"):
        dataloader = DataLoader(torch.arange(g.num_nodes()), batch_size=128,
                    shuffle=True, collate_fn=lambda indices : dgl.sampling.node2vec_random_walk(g, nodes=indices, p=args.p, q=args.q, walk_length=args.length-1))
    else:
        raise ValueError("model must be deepwalk or node2vec!")
        
    
    optimizer = SparseAdam(model.parameters(), lr=0.01)
    
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    
    logger.info("training......")
    print("training......")
    begin_train_time = time.time()
    
    for e in range(args.epoch):
        epoch_loss = 0.0
        for batch_walk in dataloader:
            loss = model(batch_walk)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalar("train loss", epoch_loss / len(dataloader), e)
        logger.info(f"epoch {e}, train loss: {epoch_loss / len(dataloader):.4f}")
    
    
           
    X = model.node_embed.weight.detach()
    y = g.ndata['label']
    clf = LogisticRegression().fit(X[train_mask].numpy(), y[train_mask].numpy())
    
    
    end_train_time = time.time()
    logger.info(f"total train time used: {end_train_time - begin_train_time:.4f}s")
    
    logger.info("testing......")
    print("testing......")
    test_acc = clf.score(X[test_mask].numpy(), y[test_mask].numpy())
    logger.info(f"test acc: {test_acc * 100:.2f}%")
        
    
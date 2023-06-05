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
    

def check_exist_and_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, help="model", default="gcn")
    parser.add_argument("-device", type=str, help="train/test device", default="cuda:0")
    parser.add_argument("-lr", type=float, help="learning rate", default=0.01)
    parser.add_argument("-hid", type=int, help="hidden feature dimension", default=16)
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
    logger = init_logging(os.path.join(log_root_with_model, hyperparam_concat(args) + ".log"))
    writer = SummaryWriter(os.path.join(tensorboard_log_root_with_model, hyperparam_concat(args)))
    device = torch.device(args.device)
    
    dataset = dgl.data.CoraGraphDataset()
    logger.info(f"Number of categories:{dataset.num_classes}")
    g = dataset[0]
    logger.info(f"graph node data: {g.ndata}")
    logger.info(f"graph edge data: {g.edata}")
    
    if (args.model == "gcn"):
        model = GCN(g.ndata["feat"].shape[1], args.hid, args.nlayer, dataset.num_classes, args.dropout)
        test_model = GCN(g.ndata["feat"].shape[1], args.hid, args.nlayer, dataset.num_classes, args.dropout)
    elif (args.model == "graphsage"):
        model = GraphSage(g.ndata["feat"].shape[1], args.hid, args.nlayer, dataset.num_classes, args.dropout)
        test_model = GraphSage(g.ndata["feat"].shape[1], args.hid, args.nlayer, dataset.num_classes, args.dropout)
    elif (args.model == "mlp"):
        model = MLP(g.ndata["feat"].shape[1], args.hid, args.nlayer, dataset.num_classes, args.dropout)
        test_model = MLP(g.ndata["feat"].shape[1], args.hid, args.nlayer, dataset.num_classes, args.dropout)
    elif (args.model == "gat"):
        model = GAT(g.ndata["feat"].shape[1], args.hid, args.nlayer, args.head, dataset.num_classes, args.dropout)
        test_model = GAT(g.ndata["feat"].shape[1], args.hid, args.nlayer, args.head, dataset.num_classes, args.dropout)
    elif (args.model == "gin"):
        model = GIN(g.ndata["feat"].shape[1], args.hid, args.nlayer, dataset.num_classes, args.dropout)
        test_model = GIN(g.ndata["feat"].shape[1], args.hid, args.nlayer, dataset.num_classes, args.dropout)
    elif (args.model == "sgc"):
        model = SGC(g.ndata["feat"].shape[1], dataset.num_classes, args.k)
        test_model = SGC(g.ndata["feat"].shape[1], dataset.num_classes, args.k)
        
    model = model.to(device)
    test_model = test_model.to(device)
    g = g.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_acc = 0
    
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    
    logger.info("training......")
    print("training......")
    begin_train_time = time.time()
    
    for e in range(args.epoch):
        logits = model(g, features)

        pred = logits.argmax(1)

        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        writer.add_scalar("train loss", loss, e)

        logger.info(f"epoch {e}, train loss: {loss:.4f}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (e % args.eval == 0):
            val_loss = F.cross_entropy(logits[val_mask], labels[val_mask])
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            writer.add_scalar("valid loss", val_loss, e)
            writer.add_scalar("valid accuracy", val_acc, e)
            logger.info("evaluating......")
            logger.info(f"epoch {e}, valid loss: {val_loss:.4f}, valid acc: {val_acc * 100:.2f}%")
            if (best_val_acc < val_acc):
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_pt_path)
    
    end_train_time = time.time()
    logger.info(f"total train time used: {end_train_time - begin_train_time:.4f}s")
    
    logger.info("testing......")
    print("testing......")
    test_model.load_state_dict(torch.load(model_pt_path))
    test_mask = g.ndata["test_mask"]
    logits = test_model(g, features)
    pred = logits.argmax(1)
    test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
    logger.info(f"test acc: {test_acc * 100:.2f}%")
        
    
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

class GraphSage(nn.Module):
    def __init__(self, in_feats, h_feats, num_layer, num_classes, dropout=0.0):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(SAGEConv(in_feats, h_feats, "mean"))
        for _ in range(num_layer - 2):
            self.layer.append(SAGEConv(h_feats, h_feats, "mean"))
        self.layer.append(SAGEConv(h_feats, num_classes, "mean"))
        self.dropout = dropout
        
    def forward(self, g, in_feat):
        h = in_feat
        for conv in self.layer[:-1]:
            h = conv(g, h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout)
        h = self.layer[-1](g, h)
        return h


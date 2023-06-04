import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_layer, num_classes, dropout=0.0):
        super(GCN, self).__init__()
        self.layer = nn.ModuleList()
        self.layer.append(GraphConv(in_feats, h_feats))
        for _ in range(num_layer - 2):
            self.layer.append(GraphConv(h_feats, h_feats))
        self.layer.append(GraphConv(h_feats, num_classes))
        self.dropout = dropout
        
    def forward(self, g, in_feat):
        h = in_feat
        for conv in self.layer[:-1]:
            h = conv(g, h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout)
        h = self.layer[-1](g, h)
        return h
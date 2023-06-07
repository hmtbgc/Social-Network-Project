import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GINConv

class GIN(nn.Module):
    def __init__(self, in_feats, h_feats, num_layer, num_classes, dropout=0.0):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(GINConv(aggregator_type="sum"))
        self.layer.append(nn.Linear(in_feats, h_feats))
        for _ in range(num_layer - 2):
            self.layer.append(GINConv(aggregator_type="sum"))
            self.layer.append(nn.Linear(h_feats, h_feats))
        self.layer.append(GINConv(aggregator_type="sum"))
        self.layer.append(nn.Linear(h_feats, num_classes))
        self.dropout = dropout
        
    def forward(self, g, in_feat):
        h = in_feat
        for conv in self.layer[:-2]:
            if (type(conv) == type(GINConv())):
                h = conv(g, h)
            else:
                h = conv(h)
                h = F.relu(h)
                h = F.dropout(h, self.dropout)
        h = self.layer[-2](g, h)
        h = self.layer[-1](h)
        return h

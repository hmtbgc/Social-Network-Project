import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_layer, num_heads, num_classes, dropout=0.0):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(GATConv(in_feats, h_feats, num_heads=num_heads))
        for _ in range(num_layer - 2):
            self.layer.append(GATConv(h_feats * num_heads, h_feats, num_heads=num_heads))
        self.layer.append(GATConv(h_feats * num_heads, num_classes, num_heads=1))
        self.dropout = dropout
        
    def forward(self, g, in_feat):
        h = in_feat
        node_number = h.shape[0]
        for conv in self.layer[:-1]:
            h = conv(g, h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout)
            h = h.view(node_number, -1)
        h = self.layer[-1](g, h)
        h = h.view(node_number, -1)
        return h
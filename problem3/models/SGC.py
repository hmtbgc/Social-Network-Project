import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SGConv
import dgl

class SGC(nn.Module):
    def __init__(self, in_feats, num_classes, k):
        super().__init__()
        self.layer = SGConv(in_feats, num_classes, k=k)
        
    def forward(self, g, in_feat):
        g = dgl.add_self_loop(g)
        return self.layer(g, in_feat)
    
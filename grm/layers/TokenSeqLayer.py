import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.SublayerConnection import *

class TokenSeqLayer(nn.Module):
    def __init__(self, emb_dim, drop_rate = 0.1):
        super(TokenSeqLayer, self).__init__()
        self.dim = emb_dim
        self.encoders = nn.ModuleList([OriAttention(drop_rate) for _ in range(1)])
        self.sublayer = SublayerConnection(drop_rate, self.dim)

    def forward(self, x, mask):
        for i, encoder in enumerate(self.encoders):
            x = self.sublayer(x, lambda x: encoder(x, mask))
        x = x.masked_fill_(~mask, 0).sum(dim=1)
        return l2norm(x)


class OriAttention(nn.Module):
    def __init__(self, drop_rate = 0.1):
        super().__init__()
        self.head = 1
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x, mask):
        q = x
        k = x
        v = x
        q, k, v = (split_last(a, (self.head, -1)).transpose(1, 2)
                   for a in [q, k, v])

        scores = torch.matmul(q, k.transpose(2, 3)) / (k.size(-1) ** 0.25)
        mask = torch.matmul(mask.float(), mask.transpose(1, 2).float()).bool()
        mask = mask.unsqueeze(1)
        mask = mask.repeat([1, self.head, 1, 1])
        scores.masked_fill_(~mask, -1e7)
        scores = F.softmax(scores, dim=2)
        scores = scores.transpose(2, 3)
        v_ = torch.matmul(scores, v)
        v_ = v_.transpose(1, 2).contiguous()
        v_ = merge_last(v_, 2)
        return v_

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def l2norm(x):
    return x / x.norm(p=2, dim=1, keepdim=True)
from layers.GCNConv import *
from utils import *

class GNNDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim ,device):
        super().__init__()
        self.device = device
        self.conv = GCNConv(hidden_dim, out_dim, 'sum')
        self.enc_to_dec = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.activation = torch.nn.PReLU()
        self.temp = 0.2


    def forward(self, x, adj, mask_node_indices, init_feature=None):
        x = self.activation(x)
        x = self.enc_to_dec(x)
        edge_index = adj._indices().to(self.device)
        if init_feature!=None:
            init_feature = init_feature.to(self.device)
            x[mask_node_indices] = init_feature
        else:
            x[mask_node_indices] = 0
        out = self.conv(x, edge_index)
        return out
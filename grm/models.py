from layers.GCNConv import *
from layers.GAT import *
from layers.ReadoutLayer import *
from loss import *
from utils import *

class R_SAGE_BN(nn.Module):
    def __init__(self, config, params, device, nfeat):
        super(R_SAGE_BN, self).__init__()
        self.isPosE = params.isPosE
        self.device = device
        self.num_layers = config['setting.num_layers']
        self.batch_size = params.batch_size
        
        self.activiation = nn.PReLU()
        self.dropout = nn.Dropout(p=0.1)
        num_of_heads = 1

        self.gnn_type = params.encoder_gnn_type

        self.bn_front = torch.nn.BatchNorm1d(nfeat)

        for index in range(1, self.num_layers+1):
            if self.gnn_type == 'gcn':
                setattr(self, 'GCNConv'+str(index), GCNConv(nfeat, nfeat))
            elif self.gnn_type == 'gat':
                setattr(self, 'GAT'+str(index), GATLayerImp3(nfeat, nfeat, num_of_heads))
            setattr(self, 'BatchNorm1d'+str(index), torch.nn.BatchNorm1d(nfeat))

        self.fc1 = nn.Linear(nfeat*(self.num_layers+1), nfeat)
        self.criterion = Align_uniform()

    def forward(self, raw_features, glyph_data, init_feature=None):
        glyph_adj = glyph_data[0]
        all_nodes = glyph_data[1]
        nodes_batch = glyph_data[2]
        features = raw_features[all_nodes]
        if init_feature!=None:
            features = features.to(self.device)
            features[nodes_batch] = init_feature
        hidden_embs = features.to(self.device)
        edge_index = glyph_adj._indices().to(self.device)
        if self.isPosE:
            edge_type = glyph_adj._values().int().to(self.device)
        else:
            edge_type = None
        hidden_embs = self.bn_front(hidden_embs)
        h_list = [hidden_embs]
        for index in range(1, self.num_layers+1):
            BatchNorm1d_layer = getattr(self, 'BatchNorm1d'+str(index))
            if self.gnn_type == 'gcn':
                GCN_layer = getattr(self, 'GCNConv'+str(index))
                if index == 1:
                    hidden_embs = GCN_layer(hidden_embs, edge_index, edge_type)
                else:
                    hidden_embs = GCN_layer(hidden_embs, edge_index)
            elif self.gnn_type == 'gat':
                GAT_layer = getattr(self, 'GAT'+str(index))
                hidden_embs,_ = GAT_layer((hidden_embs, edge_index, edge_type))

            hidden_embs = BatchNorm1d_layer(hidden_embs)
            if index != self.num_layers:
                hidden_embs = self.activiation(hidden_embs)
                hidden_embs = self.dropout(hidden_embs)
                h_list.append(hidden_embs)
            else:
                hidden_embs = self.dropout(hidden_embs)
                h_list.append(hidden_embs)

        node_representation = torch.cat(h_list, dim = 1)
        del h_list
        node_representation = self.fc1(node_representation)
        return node_representation,None

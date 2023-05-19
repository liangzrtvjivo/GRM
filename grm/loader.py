# import torch
from torch.utils.data import DataLoader
from dataCenter import WordData,OOVData
# import torch.sparse as sp
from utils import *
import time
# import torch.nn.functional as F

class SimpleLoader():
    def __init__(self, args, config, dataCenter):
        self.config = config
        self.dataCenter = dataCenter
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.raw_features = getattr(dataCenter, 'feats')
        self.nfeat = self.raw_features.shape[1]
        self.glyph_edge = getattr(dataCenter, 'glyph_edge')
        self.model_type = args.model_type
        self.char_token_list = getattr(dataCenter, 'char_token_list')
        self.piece_token_list = getattr(dataCenter, 'piece_token_list')

    def collate_fn(self, batch_data, pad=0):
        nodes_batch = batch_data
        raw_features = self.raw_features.copy()
        raw_features[nodes_batch] = 0
        raw_features = torch.from_numpy(raw_features)

        glyph_adj_batch,glyph_all_node_list,glyph_samp_nodes_batch,\
            _,raw_glyph_adj_batch = get_sample_neigh_adj(nodes_batch = nodes_batch,\
                edge_data = self.glyph_edge,\
                config = self.config, \
                is_first_sample = False)
        
        #token data
        char_token_id_list = self.char_token_list[nodes_batch]
        piece_token_id_list = self.piece_token_list[nodes_batch]

        return nodes_batch, raw_features, \
                        (glyph_adj_batch,glyph_all_node_list,glyph_samp_nodes_batch,raw_glyph_adj_batch), \
                        (char_token_id_list, piece_token_id_list)

    def __call__(self):
        dataset = WordData(self.config)
        train_iterator = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                    collate_fn=self.collate_fn)
        return train_iterator

class OOVLoader():
    def __init__(self, args, config, dataCenter, glyph_edge,char_token_list,piece_token_list,glyph_add_map):
        self.config = config
        self.dataCenter = dataCenter
        self.batch_size = args.batch_size
        oov_len = len(glyph_edge)
        self.raw_features = getattr(dataCenter, 'feats')
        self.nfeat = self.raw_features.shape[1]
        self.new_features = np.zeros(shape=(oov_len,self.nfeat), dtype=np.float32)
        self.raw_features = np.vstack((self.raw_features,self.new_features))
        self.raw_features = torch.from_numpy(self.raw_features)

        raw_glyph_edge = getattr(dataCenter, 'glyph_edge')
        raw_glyph_edge_list = raw_glyph_edge.tolist()
        for add_id,add_data in glyph_add_map.items():
            raw_glyph_edge_list[add_id] = raw_glyph_edge_list[add_id] + add_data
        glyph_edge_list = glyph_edge.tolist()
        new_glyph_edge_list = raw_glyph_edge_list + glyph_edge_list
        self.glyph_edge = np.array(new_glyph_edge_list,dtype=object)
        raw_char_token_list = getattr(dataCenter, 'char_token_list')
        raw_piece_token_list = getattr(dataCenter, 'piece_token_list')
        self.char_token_list = np.concatenate((raw_char_token_list,char_token_list),axis=0)
        self.piece_token_list = np.concatenate((raw_piece_token_list,piece_token_list),axis=0)

    def collate_fn(self, batch_data, pad=0):
        nodes_batch = batch_data
        
        glyph_adj_batch,glyph_all_node_list,glyph_samp_nodes_batch,\
            _,_ = get_sample_neigh_adj(nodes_batch = nodes_batch,\
                edge_data = self.glyph_edge,\
                config = self.config, \
                is_first_sample = False)

        #token data
        char_token_id_list = self.char_token_list[nodes_batch]
        piece_token_id_list = self.piece_token_list[nodes_batch]

        return nodes_batch, self.raw_features, \
                        (glyph_adj_batch,glyph_all_node_list,glyph_samp_nodes_batch), \
                        (char_token_id_list, piece_token_id_list)

    def __call__(self):
        dataset = OOVData(self.glyph_edge,self.config)
        train_iterator = DataLoader(dataset=dataset, batch_size=self.batch_size,
                                    collate_fn=self.collate_fn)
        return train_iterator
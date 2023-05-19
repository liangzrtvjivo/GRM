from registry import register
from functools import partial
from models import *
from layers.TokenSeqLayer import *

registry = {}
register = partial(register, registry=registry)

@register('glyph_small_bn')
class GlyphModel_small_bn(nn.Module):
    def __init__(self,config,args,device,dataCenter):
        super(GlyphModel_small_bn,self).__init__()
        n_feat = getattr(dataCenter, 'feats').shape[1]
        self.embed_model = R_SAGE_BN(config=config,
                        params=args,
                        device=device,
                        nfeat=n_feat).to(device)
        self.gnn_decoder = GNNDecoder(n_feat, n_feat, device).to(device)
        self.token_seq_layer = TokenSeqLayer(n_feat).to(device)
        self.device = device

    def forward(self,features, glyph_data, token_data):
        glyph_adj = glyph_data[0]
        nodes_batch = glyph_data[2]

        all_pred_embs,loss = self.embed_model(features, glyph_data)
        rel_hidden_embs = self.gnn_decoder(all_pred_embs, glyph_adj, nodes_batch)
        return rel_hidden_embs[nodes_batch],loss

    def infer(self, raw_features, glyph_data, token_data):
        glyph_adj = glyph_data[0]
        nodes_batch = glyph_data[2]

        all_pred_embs,_ = self.embed_model(raw_features, glyph_data)
        rel_hidden_embs = self.gnn_decoder(all_pred_embs, glyph_adj, nodes_batch)
        return rel_hidden_embs[nodes_batch]

@register('glyph_sa_1_small_bn')
class GlyphModel_sa_1_small_bn(nn.Module):
    def __init__(self,config,args,device,dataCenter):
        super(GlyphModel_sa_1_small_bn,self).__init__()
        n_feat = getattr(dataCenter, 'feats').shape[1]
        self.embed_model = R_SAGE_BN(config=config,
                        params=args,
                        device=device,
                        nfeat=n_feat).to(device)
        self.gnn_decoder = GNNDecoder(n_feat, n_feat, device).to(device)
        self.token_seq_layer = TokenSeqLayer(n_feat).to(device)
        char_num = config['setting.num_char']
        self.embedding = nn.Embedding(char_num, n_feat, padding_idx=0)
        self.embedding.weight.requires_grad = True
        self.device = device

    def forward(self,features, glyph_data, token_data):
        char_token_id_list = token_data[0]
        pad = 0
        char_token_id_tensor, char_mask = get_ids_mask(char_token_id_list,pad)
        char_mask = char_mask.to(self.device)

        total_token_id_tensor = self.embedding(char_token_id_tensor).to(self.device)
        init_feature = self.token_seq_layer(total_token_id_tensor, char_mask)

        glyph_adj = glyph_data[0]
        nodes_batch = glyph_data[2]

        all_pred_embs,loss = self.embed_model(features, glyph_data, init_feature)
        rel_hidden_embs = self.gnn_decoder(all_pred_embs, glyph_adj, nodes_batch)
        return rel_hidden_embs[nodes_batch],loss

    def infer(self, raw_features, glyph_data, token_data):
        char_token_id_list = token_data[0]
        pad = 0
        char_token_id_tensor, char_mask = get_ids_mask(char_token_id_list,pad)
        char_mask = char_mask.to(self.device)

        total_token_id_tensor = self.embedding(char_token_id_tensor).to(self.device)
        init_feature = self.token_seq_layer(total_token_id_tensor, char_mask)

        glyph_adj = glyph_data[0]
        nodes_batch = glyph_data[2]

        all_pred_embs,_ = self.embed_model(raw_features, glyph_data, init_feature)
        rel_hidden_embs = self.gnn_decoder(all_pred_embs, glyph_adj, nodes_batch)
        return rel_hidden_embs[nodes_batch]

@register('glyph_sa_1_small_no_mask_bn')
class GlyphModel_sa_1_small_no_mask_bn(nn.Module):
    def __init__(self,config,args,device,dataCenter):
        super(GlyphModel_sa_1_small_no_mask_bn,self).__init__()
        n_feat = getattr(dataCenter, 'feats').shape[1]
        self.embed_model = R_SAGE_BN(config=config,
                        params=args,
                        device=device,
                        nfeat=n_feat).to(device)
        self.gnn_decoder = GNNDecoder(n_feat, n_feat, device).to(device)
        self.token_seq_layer = TokenSeqLayer(n_feat).to(device)
        char_num = config['setting.num_char']
        self.embedding = nn.Embedding(char_num, n_feat, padding_idx=0)
        self.embedding.weight.requires_grad = True
        self.device = device

    def forward(self,features, glyph_data, token_data):
        char_token_id_list = token_data[0]
        pad = 0
        char_token_id_tensor, char_mask = get_ids_mask(char_token_id_list,pad)
        char_mask = char_mask.to(self.device)

        total_token_id_tensor = self.embedding(char_token_id_tensor).to(self.device)
        init_feature = self.token_seq_layer(total_token_id_tensor, char_mask)

        glyph_adj = glyph_data[0]
        nodes_batch = glyph_data[2]

        all_pred_embs,loss = self.embed_model(features, glyph_data, init_feature)
        rel_hidden_embs = self.gnn_decoder(all_pred_embs, glyph_adj, [])
        return rel_hidden_embs[nodes_batch],loss

    def infer(self, raw_features, glyph_data, token_data):
        char_token_id_list = token_data[0]
        pad = 0
        char_token_id_tensor, char_mask = get_ids_mask(char_token_id_list,pad)
        char_mask = char_mask.to(self.device)

        total_token_id_tensor = self.embedding(char_token_id_tensor).to(self.device)
        init_feature = self.token_seq_layer(total_token_id_tensor, char_mask)

        glyph_adj = glyph_data[0]
        nodes_batch = glyph_data[2]

        all_pred_embs,_ = self.embed_model(raw_features, glyph_data, init_feature)
        rel_hidden_embs = self.gnn_decoder(all_pred_embs, glyph_adj, [])
        return rel_hidden_embs[nodes_batch]

@register('glyph_sa_1_small_no_decoder_bn')
class GlyphModel_sa_1_small_no_decoder_bn(nn.Module):
    def __init__(self,config,args,device,dataCenter):
        super(GlyphModel_sa_1_small_no_decoder_bn,self).__init__()
        n_feat = getattr(dataCenter, 'feats').shape[1]
        self.embed_model = R_SAGE_BN(config=config,
                        params=args,
                        device=device,
                        nfeat=n_feat).to(device)
        self.token_seq_layer = TokenSeqLayer(n_feat).to(device)
        char_num = config['setting.num_char']
        self.embedding = nn.Embedding(char_num, n_feat, padding_idx=0)
        self.embedding.weight.requires_grad = True
        self.device = device

    def forward(self,features, glyph_data, token_data):
        char_token_id_list = token_data[0]
        pad = 0
        char_token_id_tensor, char_mask = get_ids_mask(char_token_id_list,pad)
        char_mask = char_mask.to(self.device)

        total_token_id_tensor = self.embedding(char_token_id_tensor).to(self.device)
        init_feature = self.token_seq_layer(total_token_id_tensor, char_mask)

        nodes_batch = glyph_data[2]

        all_pred_embs,loss = self.embed_model(features, glyph_data, init_feature)
        return all_pred_embs[nodes_batch],loss

    def infer(self, raw_features, glyph_data, token_data):
        char_token_id_list = token_data[0]
        pad = 0
        char_token_id_tensor, char_mask = get_ids_mask(char_token_id_list,pad)
        char_mask = char_mask.to(self.device)

        total_token_id_tensor = self.embedding(char_token_id_tensor).to(self.device)
        init_feature = self.token_seq_layer(total_token_id_tensor, char_mask)

        nodes_batch = glyph_data[2]

        all_pred_embs,_ = self.embed_model(raw_features, glyph_data, init_feature)
        return all_pred_embs[nodes_batch]

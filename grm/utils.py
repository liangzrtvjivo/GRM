import torch
import numpy as np
import scipy.sparse as sp
import math
import random
import codecs
from torch.nn.utils.rnn import pad_sequence
import time
import unicodedata

def get_nei_by_one_node(adj,node):
    return adj[node]._indices().tolist()[0]

def get_nei_by_list(adj,node_list):
    # print("node_list",len(node_list))
    convert_adj = sp.coo_matrix((np.ones(len(node_list),dtype=int), (node_list, node_list)), shape=adj.shape, dtype=np.float32)
    convert_adj = convert_sparse_tensor(convert_adj)
    result_adj = torch.sparse.mm(convert_adj,adj)

    indice = result_adj._indices()
    cols = list(set(indice[1].numpy()))
    # print("cols",len(cols))
    return cols,result_adj

def link_sample(adj,exist_list,is_normalize,isasymmetric,config,sample_num = 10,issample = True):
    t_begin = time.time()
    num_sub, numword = config['setting.num_sub'], config['setting.num_word']
    total_num = num_sub + numword 

    indice = adj._indices()
    value = adj._values().unsqueeze(0)
    tri_data = torch.cat([indice,value],dim=0)
    tri_data_np = tri_data.T.numpy().tolist()

    # tri_data_np = list(filter(lambda x:(x[1] not in exist_list) and (x[1] < total_num),tri_data_np))
    t_sample = time.time()
    if not issample or (tri_data.shape[1]==0):
        tri_data_list = tri_data_np
    else:
        start = tri_data_np[0][1]
        tri_data_list = []
        index = 0
        pre_index = 0
        temp_tri_data_np = tri_data_np.copy()
        for tri_data in tri_data_np:
            if tri_data[0]!=start:
                start = tri_data[0]
                tri_temp_list = temp_tri_data_np[pre_index:index]
                random.shuffle(tri_temp_list)
                tri_data_list += tri_temp_list[:min(sample_num,len(tri_temp_list))]
                pre_index = index
            if (tri_data[1] not in exist_list) and (tri_data[1] < total_num):
                index += 1
            else:
                del temp_tri_data_np[index]
    sample_tri_data = np.array(tri_data_list).T
    # sample_indice = torch.tensor(sample_tri_data[:1],dtype=torch.int64)
    # sample_value = sample_tri_data[2].squeeze(0)
    # nei_list = list(set(sample_tri_data[1].squeeze(0)))
    # print("sample_indice",sample_indice)
    # print("sample_value",sample_value)
    # sample_adj = torch.sparse_coo_tensor(sample_indice, sample_value, adj.shape)
    if len(sample_tri_data)!=0:
        row_value = np.int64(sample_tri_data[0]).tolist()
        col_value = np.int64(sample_tri_data[1]).tolist()
        link_value = sample_tri_data[2].tolist()
    else:
        row_value = []
        col_value = []
        link_value = []

    if isasymmetric:
        sample_adj = sp.coo_matrix((link_value, (row_value, col_value)), shape=adj.shape, dtype=np.float32)
    else:
        sample_adj = sp.coo_matrix((link_value+link_value, (row_value+col_value, col_value+row_value)), shape=adj.shape, dtype=np.float32)
    if is_normalize:
        sample_adj = normalize(sample_adj)
    sample_adj = convert_sparse_tensor(sample_adj)

    nei_list = list(set(col_value))
    t_construct = time.time()
    # print("util.py t_construct time elapsed: {:.4f}s".format(t_construct - t_sample))
    return sample_adj,nei_list

def link_dropout(adj, config, mask_num=5):
    mask_occur_adj,_ = link_sample(adj = adj,exist_list = [],is_normalize = True,isasymmetric = True,config = config,\
                        sample_num = mask_num)

    return mask_occur_adj

def adj_map(adj,node_list,nodes_batch):
    node_idx_list = {}
    index = 0
    for node in node_list:
        node_idx_list[node] = index
        index+=1
    
    indice = adj._indices()
    rows = indice[0].numpy()
    cols = indice[1].numpy()
    value = adj._values().numpy()

    samp_rows = [node_idx_list[node] for node in rows]
    samp_cols = [node_idx_list[node] for node in cols]
    samp_nodes_batch = [node_idx_list[node] for node in nodes_batch]
    
    sample_adj = sp.coo_matrix((value, (samp_rows, samp_cols)), shape=(len(node_list),len(node_list)), dtype=np.float32)
    sample_adj = convert_sparse_tensor(sample_adj)
    return sample_adj,samp_nodes_batch

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = np.where(rowsum==0, 1, rowsum)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)    
    mx = r_mat_inv.dot(mx)

    return mx
    
def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def convert_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def construct_adj(e, shape, is_weight, isasymmetric=False):
    if not is_weight:
        adj = sp.coo_matrix((np.ones((e.shape[0]), dtype=np.float32), (e[:, 0], e[:, 1])), shape=shape)
    else:
        adj = sp.coo_matrix((e[:, 2], (e[:, 0], e[:, 1])), shape=shape, dtype=np.float32)
    #adj += adj.transpose()
    if not isasymmetric:
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj

def add_self_loop_for_sparse(adj):
    oov_list = [i for i in range(0,adj.shape[0])]
    oov_indices = torch.from_numpy(np.array([oov_list,oov_list]))
    indices = torch.cat((adj._indices(),oov_indices),dim=1)
    values = torch.cat((adj._values(),torch.full([adj.shape[0]],3)),dim=0) #word token:3
    adj = torch.sparse_coo_tensor(indices, values, adj.shape)
    return adj

def get_sample_nei_by_list(edge_data,node_list,sample_num,exist_list,config,isasymmetric,is_normalize,issample = True):
    t_begin = time.time()
    nei_node_list = edge_data[node_list].tolist()
    t_225 = time.time()
    # print("get_sample_nei_by_list t_225 time elapsed: {:.4f}s".format(t_225 - t_begin))
    num_sub, numword = config['setting.num_sub'], config['setting.num_word']
    raw_total_num = num_sub + numword 
    total_num = len(edge_data)
    sample_node_list = []
    for nei_node_data in nei_node_list:
        temp = [i for i in nei_node_data if i[1] not in exist_list and i[1] < raw_total_num]
        if issample:
            random.shuffle(temp)
            temp = temp[:min(sample_num,len(temp))]
        sample_node_list = sample_node_list + temp
    t_236 = time.time()
    # print("get_sample_nei_by_list t_236 time elapsed: {:.4f}s".format(t_236 - t_225))
    sample_array = np.array(sample_node_list).T
    if len(sample_array)==0:
        row_value = []
        col_value = []
        link_value = []
    else:
        row_value = np.int64(sample_array[0]).tolist()
        col_value = np.int64(sample_array[1]).tolist()
        link_value = sample_array[2].tolist()

    if isasymmetric:
        sample_adj = sp.coo_matrix((link_value, (row_value, col_value)), shape=(total_num,total_num), dtype=np.float32)
    else:
        sample_adj = sp.coo_matrix((link_value+link_value, (row_value+col_value, col_value+row_value)), shape=(total_num,total_num), dtype=np.float32)
    if is_normalize:
        sample_adj = normalize(sample_adj)
    sample_adj = convert_sparse_tensor(sample_adj)
    nei_list = list(set(col_value))
    t_250 = time.time()
    # print("get_sample_nei_by_list t_250 time elapsed: {:.4f}s".format(t_250 - t_236))
    return sample_adj,nei_list

def get_sample_neigh_adj(nodes_batch, edge_data, config, isasymmetric = False, is_first_sample = True):
    # edge_data store tri-data
    first_nei_samp = config['proc_setting.first_nei_samp']
    second_nei_samp = config['proc_setting.second_nei_samp']
    t_begin = time.time()

    one_nei_adj,one_nei_list = get_sample_nei_by_list(edge_data = edge_data,node_list = nodes_batch,sample_num = first_nei_samp,\
                                            exist_list = nodes_batch,config = config,\
                                            isasymmetric = isasymmetric,is_normalize = isasymmetric,\
                                            issample = is_first_sample)
    t_175 = time.time()
    # print("get_sample_neigh_adj t_175 time elapsed: {:.4f}s".format(t_175 - t_begin))

    two_nei_adj,two_nei_list = get_sample_nei_by_list(edge_data = edge_data,node_list = one_nei_list,sample_num = second_nei_samp,\
                                            exist_list = list(set(nodes_batch+one_nei_list)),config = config,\
                                            isasymmetric = isasymmetric,is_normalize = isasymmetric)
    t_184 = time.time()
    # print("get_sample_neigh_adj t_184 time elapsed: {:.4f}s".format(t_184 - t_175))
    
    all_list = list(set(nodes_batch+one_nei_list+two_nei_list))
    raw_result_adj = one_nei_adj + two_nei_adj

    result_adj,samp_nodes_batch = adj_map(raw_result_adj,all_list,nodes_batch)
    t_191 = time.time()
    # print("get_sample_neigh_adj t_191 time elapsed: {:.4f}s".format(t_191 - t_184))
    result_mask_adj = None

    #add self loop
    result_adj = add_self_loop_for_sparse(result_adj)

    return result_adj,all_list,samp_nodes_batch,result_mask_adj,raw_result_adj


# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

""" Tokenization classes (It's exactly the same code as Google BERT code """

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

def _run_split_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]

def tokenize(text, piece_list):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """
        unk_token = "[UNK]"
        max_input_chars_per_word = 100
        text = text.lower()
        token = _run_strip_accents(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            split_tokens.extend(_run_split_on_punc(token))

        # text = convert_to_unicode(text)

        output_tokens = []
        for token in split_tokens:
            chars = list(token)
            if len(chars) > max_input_chars_per_word:
                output_tokens.append(unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in piece_list:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

def process_oov_file_by_piece(oov_file,config):
    wiki_word2id = config['file_path.wiki_word2id']
    wiki_piece2id = config['file_path.wiki_piece2id']

    num_sub, numword = config['setting.num_sub'], config['setting.num_word']

    pieceid_list = {}
    with codecs.open(wiki_piece2id, "r", "utf-8") as f:
        for i, line in enumerate(f):
            lines = line.rstrip().split(" ")
            piece = lines[1]
            pieceid = int(lines[0])
            pieceid_list[piece] = pieceid
    w2id_list = {}
    with codecs.open(wiki_word2id, "r", "utf-8") as f:
        for i, line in enumerate(f):
            lines = line.rstrip().split(" ")
            word = lines[1]
            wid_t = int(lines[0])
            wid = wid_t + num_sub
            w2id_list[word] = wid

    with codecs.open(oov_file, "r", "utf-8") as f:
        oov_id_map = {}
        oov_word_map = {}
        print("convert word to id and generate word-occur_word-rate dictionary")
        for i, line in enumerate(f):
            lines = line.rstrip().split("\t")
            oov_word = lines[0]
            if oov_word not in oov_id_map:
                oov_id = num_sub + numword + len(oov_id_map)
                oov_id_map[oov_word] = oov_id
                oov_word_map[oov_id] = oov_word
            else:
                oov_id = oov_id_map[oov_word]
        
        glyph_edge_list = [[] for i in range(len(oov_id_map))]
        char_token_add_list = [[] for i in range(len(oov_id_map))]
        piece_token_add_list = [[] for i in range(len(oov_id_map))]
        glyph_add_map = {}
        for oov_word,oov_id in oov_id_map.items():
            chars = list(oov_word)
            pieces = tokenize(oov_word, pieceid_list)
            pi_len = len(pieces)
            if pi_len==1:
                pi_idx = 4
            else:
                pi_idx = 5
            idx = oov_id - (num_sub + numword)
            for char in chars:
                c_id = pieceid_list[char] if char in pieceid_list else pieceid_list['[UNK]']
                char_token_add_list[idx].append(c_id)

            for piece in pieces:
                p_id = pieceid_list[piece] if piece in pieceid_list else pieceid_list['[UNK]']
                glyph_edge_list[idx].append((oov_id,p_id,pi_idx))
                piece_token_add_list[idx].append(p_id)
                if p_id not in glyph_add_map:
                    glyph_add_map[p_id] = []
                glyph_add_map[p_id].append((p_id,oov_id,pi_idx))
                pi_idx+=1
        
        glyph_edge = np.array(glyph_edge_list,dtype=object)
        char_token_list = np.array(char_token_add_list,dtype=object)
        piece_token_list = np.array(piece_token_add_list,dtype=object)
        
    return glyph_edge,char_token_list,piece_token_list,glyph_add_map,oov_word_map

def get_random_augment_word(node,syn_list,sim_list,probs=[0.3,0.3,0.4]):
    random_type = ['syn','sim','unchange']
    if len(syn_list)==0:
        probs[0] = 0
    if len(sim_list)==0:
        probs[1] = 0

    random_probs = np.array(probs)
    if sum(random_probs)==0:
        return node
    random_probs = random_probs / sum(random_probs)
    random_result = np.random.choice(random_type, 1, p=random_probs)[0]
    if random_result == 'syn':
        return np.random.choice(syn_list, 1)[0]
    elif random_result == 'sim':
        return np.random.choice(sim_list, 1)[0]
    else:
        return node

def get_data_augment(nodes_batch,adj,config,dataCenter,raw_features,probs = [0.3,0.3,0.4]):
    num_sub, numword = config['setting.num_sub'], config['setting.num_word']
    word_syn = getattr(dataCenter, 'word_syn')
    char_index = num_sub
    aug_word_batch = []
    for node in nodes_batch:
        # can be split into common wordpiece and syn wordpiece
        one_neigh_list = get_nei_by_one_node(adj,node)
        # syn_wordpiece_index = 21257 #special token index
        # common_piece_list = [common for common in one_neigh_list if common<syn_wordpiece_index]
        # syn_piece_list = [syn for syn in one_neigh_list if syn>=syn_wordpiece_index]
        common_piece_list = one_neigh_list
        # if len(syn_piece_list)>0:
        #     two_syn_neigh_list,_ = get_nei_by_list(adj,syn_piece_list)
        #     syn_list = [syn for syn in two_syn_neigh_list if syn >= char_index and syn != node]
        # else:
        #     syn_list = []
        syn_list = word_syn[node]
        
        two_common_neigh_list,_ = get_nei_by_list(adj,common_piece_list)
        # the word contains the same char with original word
        sim_list = [sim for sim in two_common_neigh_list if sim >= char_index and sim != node]
        aug_word = get_random_augment_word(node,syn_list,sim_list,probs)
        aug_word_batch.append(aug_word)
    return raw_features[aug_word_batch]

def get_ids_mask(id_list,pad):
    max_len = max([len(seq) for seq in id_list])
    id_tensor = [char + [pad] * (max_len - len(char)) for char in id_list]
    id_tensor = torch.LongTensor(id_tensor)
    mask = torch.ne(id_tensor, pad).unsqueeze(2)
    return id_tensor,mask

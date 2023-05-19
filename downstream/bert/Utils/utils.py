# coding=utf-8

import time
from sklearn import metrics
import random
import numpy as np
import gensim

import torch
import torch.nn as nn

def get_device(gpu_id):
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")
    return device, n_gpu

def load_new_embedding(ctrlname,input_file,input_name):
    vectors_path = '../../../grm/result/resultfile_' + ctrlname  + '/oov_result'+ input_name + input_file +'.txt'
    model = gensim.models.KeyedVectors.load_word2vec_format(vectors_path)

    new_embeddings = model.vectors
    words = list(model.key_to_index.keys())
    print("words",len(words))
    print("new_embeddings",new_embeddings.shape)
    process_words = ['%%'+word for word in words]

    return words,new_embeddings,process_words

def reset_bert_embedding(model,raw_embeddings):
    old_embeddings = model.get_input_embeddings()
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    new_num_tokens = old_num_tokens + raw_embeddings.shape[0]
    new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
    new_embeddings.weight.data[:old_num_tokens, :] = old_embeddings.weight.data[:old_num_tokens, :]
    new_embeddings.weight.data[old_num_tokens:, :] = torch.tensor(raw_embeddings)[:,:]
    model.set_input_embeddings(new_embeddings)

def get_oov_label(labels,isoovs):
    oov_label = []
    for label,isoov in zip(labels,isoovs):
        if isoov == 1:
            oov_label.append(label)
    return oov_label

def get_unmask_label(labels,masks,wids):
    m_label = []
    m_wid = []
    for label,mask,wid in zip(labels,masks,wids):
        if mask == 1:
            m_label.append(label)
            m_wid.append(wid)
    return m_label,m_wid

def merge_token(wids,labels):
    first_label = -1
    pre_w = 0
    all_same = False
    total_label_list = []
    oov_label_list = []
    for wid,label in zip(wids,labels):
        if wid==0:
            if pre_w != 0:
                w_label = deal_pre_word_label(all_same,first_label)
                total_label_list.append(w_label)
                oov_label_list.append(w_label)
            pre_w = 0
            total_label_list.append(label)
        else:
            if wid!=pre_w:
                w_label = deal_pre_word_label(all_same,first_label)
                total_label_list.append(w_label)
                oov_label_list.append(w_label)

                wid = pre_w #begin of an oov word
                first_label = label
                all_same = True
            else:
                if label != first_label:
                    all_same = False
    return total_label_list,oov_label_list

def deal_pre_word_label(all_same,first_label):
    false_label = -1
    if all_same:
        return first_label
    else:
        return false_label


def merge_token_label(batch_w2t_list,batch_label_list,batch_isoov):
    false_label = -1
    all_origin_label_list = []
    all_oov_label_list = []
    for w2t_list,label_list,isoov in zip(batch_w2t_list,batch_label_list,batch_isoov):
        origin_label_list = []
        for t_list in w2t_list:
            t_labels = [label_list[t] for t in t_list]
            if len(t_labels)==1:
                origin_label_list = origin_label_list + t_labels
            else:
                #check if all label is the same
                first_label = t_labels[0]
                all_same = True
                for t_label in t_labels:
                    if t_label != first_label:
                        all_same = False
                if all_same:
                    origin_label_list.append(first_label)
                else:
                    origin_label_list.append(false_label)
        oov_label_list = []
        for index in range(len(origin_label_list)):
            if isoov[index] == 1:
                oov_label_list.append(origin_label_list[index])
        all_origin_label_list+=origin_label_list
        all_oov_label_list+=oov_label_list
    return all_origin_label_list,all_oov_label_list
import random
import numpy as np
import os

import torch
import torch.nn as nn
import argparse

from transformers import BertTokenizerFast,BertForTokenClassification
from transformers import AdamW
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.utils import *
from dataprocess import load_pos_data
from utils import *
from evaluate_morphotags import *
import collections

Instance = collections.namedtuple("Instance", ["sentence", "tags"])

parser = argparse.ArgumentParser(description='BERT Baseline')
parser.add_argument("--model_name", default="BertLinear", type=str, help="the name of model")
parser.add_argument("--save_name", default="BertLinear",type=str, help="the name file of model")
parser.add_argument("--data_dir",
                    default="./ud_1.4/ud_en_1.4",
                    type=str,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--log_dir",
                    default="sst_log/BertLinear",
                    type=str,
                    help="日志目录，主要用于 tensorboard 分析")
parser.add_argument("--bert_vocab_file",
                        default='../Pretrained/bert-base-uncased',
                        type=str)
parser.add_argument("--bert_model_dir",
                        default='../Pretrained/bert-base-uncased',
                        type=str)
parser.add_argument('--seed',
                    type=int,
                    default=49,
                    help="随机种子 for initialization")
parser.add_argument("--do_lower_case",
                    default=True,
                    type=bool,
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
parser.add_argument("--batch_size",
                    default=16,
                    type=int)
parser.add_argument("--num_train_epochs",
                    default=5,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--learning_rate", 
                    default=5e-5,
                    type=float,
                    help="Adam 的 学习率")
parser.add_argument('--print_step',
                    type=int,
                    default=50,
                    help="多少步进行模型保存以及日志信息写入")
parser.add_argument("--gpu_ids", type=str, default="0", help="gpu 的设备id")
parser.add_argument('--add_new', type=bool, default=False)
parser.add_argument('--ctrlname', type=str, default='2022-11-10-20_16_32')
parser.add_argument('--input_file', type=str, default='5')
parser.add_argument('--input_name', type=str, default='_ud_1.4_')
parser.add_argument('--mimick_type', help='replace, origin, linear_combine', type=str, default='replace')

def train(model,train_loader,optim,epoch):
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch in train_loader:
        # 正向传播
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        isoov = batch['isoov'].to(device)
        wids = batch['wids'].cpu().numpy()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()
        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 参数更新
        optim.step()
        iter_num += 1
        if(iter_num % 100==0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (epoch, iter_num, loss.item(), iter_num/total_iter*100))
    print("Epoch: %d, Average training loss: %.4f"%(epoch, total_train_loss/len(train_loader)))

def validation(model,dataloader,num_labels):
    model.eval()
    total_eval_loss = 0
    total_gold_label = []
    total_pre_label = []
    oov_gold_label = []
    oov_pre_label = []
    for batch in dataloader:
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            isoov = batch['isoov'].to(device)
            wids = batch['wids'].cpu().numpy()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        pre_label = np.argmax(logits, axis=2).flatten().tolist()
        gold_label = label_ids.flatten().tolist()
        isoovs = isoov.to('cpu').numpy().flatten().tolist()
        widss = wids.flatten().tolist()
        masks = attention_mask.to('cpu').numpy().flatten().tolist()
        pre_um_labels,pre_um_widss = get_unmask_label(pre_label,masks,widss)
        gold_um_labels,gold_um_widss = get_unmask_label(gold_label,masks,widss)
        t_pre_label,o_pre_label = merge_token(pre_um_widss,pre_um_labels)
        t_gold_label,o_gold_label = merge_token(gold_um_widss,gold_um_labels)

        total_gold_label += t_gold_label
        total_pre_label += t_pre_label
        oov_gold_label += o_gold_label
        oov_pre_label += o_pre_label
        total_eval_loss += loss
        
    acc,rec,f1 = evaluate(total_gold_label,total_pre_label,num_labels)
    o_acc,o_rec,o_f1 = evaluate(oov_gold_label,oov_pre_label,num_labels)
    print("-------------------------------")
    print(f"Accuracy: {acc} Recall: {rec} F1: {f1} ")
    print(f"OOV Accuracy: {o_acc} Recall: {o_rec} F1: {o_f1} ")
    print("Average testing loss: %.4f"%(total_eval_loss/len(dataloader)))
    print("-------------------------------")

if __name__ == '__main__':
    config = parser.parse_args()
    # 设备准备
    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
    device, n_gpu = get_device(gpu_ids[0])  
    # 设定随机种子 
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)# 数据准备
    words,new_embeddings,process_words = load_new_embedding(config.ctrlname,config.input_file,config.input_name)

    tokenizer = BertTokenizerFast.from_pretrained(
        config.bert_vocab_file, do_lower_case=config.do_lower_case) 
    if config.add_new:
        #add oov as new token
        num_added_toks = tokenizer.add_tokens(process_words)
    dataloader_list,pos_i2t,attributes = load_pos_data(
        config.data_dir, tokenizer, config.max_seq_length, config.batch_size, config.add_new, words, process_words)
    num_labels = len(attributes)
    train_dataloader = dataloader_list[0]
    dev_dataloader = dataloader_list[1]
    test_dataloader = dataloader_list[2]

    model = BertForTokenClassification.from_pretrained(config.bert_model_dir, num_labels=num_labels)
    if config.add_new:
        reset_bert_embedding(model,new_embeddings)
    device = torch.device("cuda:%d" % gpu_ids[0] if torch.cuda.is_available() else "cpu")
    model.to(device)    

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                            lr=config.learning_rate)

    for epoch in range(int(config.num_train_epochs)):
        train(model,train_dataloader,optimizer,epoch)
        validation(model,dev_dataloader,num_labels)

    validation(model,test_dataloader,num_labels)
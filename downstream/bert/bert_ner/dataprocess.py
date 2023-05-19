import os
from numpy import add
import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.dataloader import default_collate

class NerDataset(Dataset):
    def __init__(self, encodings, labels, isoov, wids):
        self.encodings = encodings
        self.labels = labels
        self.isoov = isoov
        self.wids = wids

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['isoov'] = torch.tensor(self.isoov[idx])
        item['wids'] = torch.tensor(self.wids[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_txt_ner_dataset(filename, set_type, add_new, oov_words):
    labels = []
    texts = []
    isoovs = []
    text, label, isoov = [], [], []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0].lower()
                if word in oov_words:
                    isoov.append(1)
                    if add_new :
                        word = '%%' + word
                else:
                    isoov.append(0)
                text.append(word)
                label.append(pairs[-1])
            else:
                if len(text) > 0:
                    texts.append(' '.join(text))
                    labels.append(label.copy())
                    isoovs.append(isoov.copy())
                text, label, isoov = [], [], []
    return texts,labels,isoovs

def load_label(data_dir):
    file_list = [
        os.path.join(data_dir, 'train.txt'),
        os.path.join(data_dir, 'dev.txt'),
        os.path.join(data_dir, 'test.txt'),
    ]
    label_list = {}
    num_label = 0
    label_id2en = {}
    label_id2en[-1] = 'DEF-FALSE'
    for filename in file_list:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    label = pairs[-1]
                    if label not in label_list:
                        label_list[label] = num_label
                        label_id2en[num_label] = label
                        num_label += 1
    return label_list,label_id2en

def load_ner_data(data_dir, tokenizer, max_length, batch_size, data_type, label_map, add_new, oov_words, process_words):
    load_func = load_txt_ner_dataset
    if not add_new:
        process_words = oov_words
    if data_type == "train":
        train_file = os.path.join(data_dir, 'train.txt')
        examples = load_func(train_file, data_type, add_new, oov_words)
    elif data_type == "dev":
        dev_file = os.path.join(data_dir, 'dev.txt')
        examples = load_func(dev_file, data_type, add_new, oov_words)
    elif data_type == "test":
        test_file = os.path.join(data_dir, 'test.txt')
        examples = load_func(test_file, data_type, add_new, oov_words)
    else:
        raise RuntimeError("should be train or dev or test")

    texts,labels,isoovs = examples

    enc = tokenizer("london", truncation=True, padding=True, max_length=max_length)
    print("enc",enc)
    enc = tokenizer("-docstart-", truncation=True, padding=True, max_length=max_length)
    print("enc1",enc)

    encoding = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_offsets_mapping=True)
    # enc = tokenizer("london", truncation=True, padding=True, max_length=max_length, return_offsets_mapping=True)
    # print("enc",enc)
    # encoding = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    
    all_label_list = []
    all_isoov_list = []
    all_wids_list = []
    index = 0
    for index in range(len(texts)):
        label_list = []
        isoov_list = []
        wids_list = []
        text_list = texts[index].split()
        tokens = encoding.tokens(index)
        text_idx = 0
        t = text_list[text_idx].lower()
        comple_t = t
        token_idx = 0
        for token in tokens:
            token = token.replace("##","")
            if token in t:
                l = labels[index][text_idx]
                i_o = isoovs[index][text_idx]
                label_list.append(label_map[l])
                isoov_list.append(i_o)
                if i_o == 1:
                    oov_t_idx = process_words.index(comple_t)
                    wids_list.append(oov_t_idx)
                else:
                    wids_list.append(0)

                t = t[len(token):]
                if len(t) == 0:
                    text_idx += 1
                    if text_idx < len(text_list):
                        t = text_list[text_idx].lower()
                        comple_t = t
                    else:
                        t = ''
            else:
                label_list.append(label_map['O'])
                isoov_list.append(0)
                wids_list.append(0)
            token_idx += 1
        if index < 3:
            print("texts",texts[index])
            print("ids",encoding["input_ids"][index])
            print("mask",encoding["attention_mask"][index])
            print("tokens",tokens)
            print("label_list",label_list)
        all_label_list.append(label_list)
        all_isoov_list.append(isoov_list)
        all_wids_list.append(wids_list)
        
    dataset = NerDataset(encoding, all_label_list,all_isoov_list,all_wids_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
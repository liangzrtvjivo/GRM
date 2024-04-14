import os
import torch
from torch.utils.data import Dataset,DataLoader
import random
import pickle

NONE_TAG = "<NONE>"

class PosDataset(Dataset):
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

def load_txt_pos_dataset(filename, add_new, oov_words):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    w2i = dataset["w2i"]
    t2is = dataset["t2is"]
    c2i = dataset["c2i"]
    pos_t2i = t2is["POS"]
    i2w = { i: w for w, i in list(w2i.items()) } # Inverse mapping
    i2ts = { att: {i: t for t, i in list(t2i.items())} for att, t2i in list(t2is.items()) }
    i2c = { i: c for c, i in list(c2i.items()) }
    pos_i2t = i2ts["POS"]

    training_instances = dataset["training_instances"]
    dev_instances = dataset["dev_instances"]
    test_instances = dataset["test_instances"]

    attributes = list(pos_t2i.keys())
    # print("attributes",attributes)

    train_labels = []
    train_texts = []
    train_isoov = []
    for instance in training_instances:
        if len(instance.sentence) == 0: continue
        gold_tags = instance.tags["POS"]
        text_list = instance.sentence
        word_list = []
        isoov = []
        for text in text_list:
            word = i2w[text].lower()
            if word in oov_words:
                isoov.append(1)
                if add_new :
                    word = '%%' + word
            else:
                isoov.append(0)
            word_list.append(word)
        text = ' '.join(word_list)
        train_texts.append(text)
        train_labels.append(gold_tags)
        train_isoov.append(isoov)

    dev_labels = []
    dev_texts = []
    dev_isoov = []
    for instance in dev_instances:
        if len(instance.sentence) == 0: continue
        gold_tags = instance.tags["POS"]
        text_list = instance.sentence
        word_list = []
        isoov = []
        for text in text_list:
            word = i2w[text].lower()
            if word in oov_words:
                isoov.append(1)
                if add_new :
                    word = '%%' + word
            else:
                isoov.append(0)
            word_list.append(word)
        text = ' '.join(word_list)
        dev_texts.append(text)
        dev_labels.append(gold_tags)
        dev_isoov.append(isoov)

    test_labels = []
    test_texts = []
    test_isoov = []
    for instance in test_instances:
        if len(instance.sentence) == 0: continue
        gold_tags = instance.tags["POS"]
        text_list = instance.sentence
        word_list = []
        isoov = []
        for text in text_list:
            word = i2w[text].lower()
            if word in oov_words:
                isoov.append(1)
                if add_new :
                    word = '%%' + word
            else:
                isoov.append(0)
            word_list.append(word)
        text = ' '.join(word_list)
        test_texts.append(text)
        test_labels.append(gold_tags)
        test_isoov.append(isoov)
    return (train_labels,train_texts,train_isoov),(dev_labels,dev_texts,dev_isoov),(test_labels,test_texts,test_isoov),(pos_i2t,attributes)

def load_pos_data(data_dir, tokenizer, max_length, batch_size, add_new, oov_words, process_words):
    result = load_txt_pos_dataset(data_dir, add_new, oov_words)
    if not add_new:
        process_words = oov_words

    pos_i2t,attributes = result[3]
    dataloader_list = []
    for examples in result[:3]:
        labels,texts,isoov = examples
        encoding = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_offsets_mapping=True)
    
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
            for token in tokens:
                token = token.replace("##","")
                if token in t:
                    l = labels[index][text_idx]
                    i_o = isoov[index][text_idx]
                    label_list.append(l)
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
                    label_list.append(0)
                    isoov_list.append(0)
                    wids_list.append(0)
            # if index < 3:
            #     print("texts",texts[index])
            #     print("isoov",isoov[index])
            #     print("ids",encoding["input_ids"][index])
            #     print("tokens",tokens)
            #     print("label_list",label_list)
            #     print("isoov_list",isoov_list)
            #     print("wids_list",wids_list)
            all_label_list.append(label_list)
            all_isoov_list.append(isoov_list)
            all_wids_list.append(wids_list)

        dataset = PosDataset(encoding, all_label_list, all_isoov_list, all_wids_list)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloader_list.append(dataloader)

    return dataloader_list,pos_i2t,attributes
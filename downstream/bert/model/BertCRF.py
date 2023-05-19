# coding=utf-8

from transformers import BertForTokenClassification

import torch
from torch import nn
from TorchCRF import CRF
from torch.nn.utils.rnn import pad_sequence

class BertCRF(nn.Module):
    def __init__(self, config, num_labels):
        super(BertCRF, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(config, num_labels=num_labels)
        self.crf = CRF(num_labels=num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        if labels is not None:
            bool_mask = (attention_mask > 0)
            loss = -self.crf(logits, labels, bool_mask)
            tags = self.crf.viterbi_decode(logits, bool_mask)
            tag_list = []
            shape_len = attention_mask.shape[1]
            for tag in tags:
                tag = tag + [0 for i in range(shape_len-len(tag))]
                tag_list.append(torch.tensor(tag).unsqueeze(0))
            tags_tensor = pad_sequence(tag_list,padding_value=0)
        return loss.mean(),tags_tensor
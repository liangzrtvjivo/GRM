from pickle import TRUE
# import sys
import os

# from collections import defaultdict
import numpy as np
from utils import *
# from sklearn.utils import shuffle
from torch.utils.data import Dataset
# import tokenization

class WordData(Dataset):
    def __init__(self, config):
        self.config = config
        self.num_sub, self.numword = \
					config['setting.num_sub'], config['setting.num_word']

    def __len__(self):
        return self.numword

    def __getitem__(self, idx):
        return self.num_sub + idx

class OOVData(Dataset):
    def __init__(self, glyph_edge, config):
        self.glyph_edge = glyph_edge
        self.config = config
        self.num_sub, self.numword = \
					config['setting.num_sub'], config['setting.num_word']

    def __len__(self):
        return len(self.glyph_edge) - (self.num_sub + self.numword)

    def __getitem__(self, idx):
        oov_idx = self.num_sub + self.numword + idx
        return oov_idx

class DataCenter(object):
	"""docstring for DataCenter"""
	def __init__(self, config):
		super(DataCenter, self).__init__()
		self.config = config

	def load_dataSet(self):
		self.split_type = self.config['setting.split_type']
		self.load_en_wiki()

	def load_en_wiki(self):
		print("Load start")
		if self.split_type == 2:
			self.sub_num = self.config['setting.num_piece']
		self.config.put('setting.num_sub',self.sub_num)
		feat_data = self.load_en_feat()
		print("Load feat done")
		glyph_edge = self.load_en_glyph_npy()
		print("Load glyph adj done")
		char_token_list = self.load_char_token_npy()
		piece_token_list = self.load_piece_token_npy()
		print("Load token done")
		word_syn = self.load_syn_npy()
		print("Load syn done")
		
		assert len(feat_data) == len(glyph_edge)

		setattr(self, 'feats', feat_data)
		setattr(self, 'glyph_edge', glyph_edge)
		setattr(self, 'char_token_list', char_token_list)
		setattr(self, 'piece_token_list', piece_token_list)
		setattr(self, 'word_syn', word_syn)

	def load_en_feat(self):
		if os.path.exists(self.config['file_path.wiki_feat_file_npy']):
			feat_data = np.load(self.config['file_path.wiki_feat_file_npy'])
		else:
			feat_data = [] #feat is embedding
			if self.split_type == 2:
				subword_feature_file = self.config['file_path.wiki_pieceid2feature']
			with open(subword_feature_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					feat_data.append([float(x) for x in info[1:]])
			word_feature_file = self.config['file_path.wiki_wid2feature']
			with open(word_feature_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					feat_data.append([float(x) for x in info[1:]])
			feat_data = np.asarray(feat_data, dtype=np.float32)
			np.save(self.config['file_path.wiki_feat_file_npy'],feat_data)
		return feat_data

	def load_char_token_npy(self):
		if os.path.exists(self.config['file_path.wiki_char_token_file_npy']):
			char_token_list = np.load(self.config['file_path.wiki_char_token_file_npy'],allow_pickle=True)
		else:
			num_sub, numword = \
						self.config['setting.num_sub'], self.config['setting.num_word']
			raw_token_list = [[] for i in range(num_sub + numword)]

			wiki_wid2char_file = self.config['file_path.wiki_wid2char']
			with open(wiki_wid2char_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					assert len(info) == 3
					paper1 = int(info[0])+num_sub # word idx
					paper2 = int(info[1])
					raw_token_list[paper1].append(paper2)

			char_token_list = np.array(raw_token_list,dtype=object)
			del raw_token_list
			np.save(self.config['file_path.wiki_char_token_file_npy'],char_token_list)

		return char_token_list

	def load_piece_token_npy(self):
		if os.path.exists(self.config['file_path.wiki_piece_token_file_npy']):
			piece_token_list = np.load(self.config['file_path.wiki_piece_token_file_npy'],allow_pickle=True)
		else:
			num_sub, numword = \
						self.config['setting.num_sub'], self.config['setting.num_word']
			raw_token_list = [[] for i in range(num_sub + numword)]

			if self.split_type == 2:
				wiki_relation_file = self.config['file_path.wiki_wid2pieceid']
			with open(wiki_relation_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					assert len(info) == 3
					paper1 = int(info[0])+num_sub # word idx
					paper2 = int(info[1])
					raw_token_list[paper1].append(paper2)

			piece_token_list = np.array(raw_token_list,dtype=object)
			del raw_token_list
			np.save(self.config['file_path.wiki_piece_token_file_npy'],piece_token_list)

		return piece_token_list

	def load_syn_npy(self):
		if os.path.exists(self.config['file_path.wiki_syn_file_npy']):
			word_syn = np.load(self.config['file_path.wiki_syn_file_npy'],allow_pickle=True)
		else:
			num_sub, numword = \
						self.config['setting.num_sub'], self.config['setting.num_word']
			word_syn_list = [[] for i in range(num_sub + numword)]
			wiki_word_syn_file = self.config['file_path.wiki_word_syn_file']
			with open(wiki_word_syn_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					assert len(info) == 3
					paper1 = int(info[0])+num_sub # word idx
					paper2 = int(info[1])+num_sub # word idx
					word_syn_list[paper1].append(paper2)
					word_syn_list[paper2].append(paper1)
			word_syn = np.array(word_syn_list,dtype=object)
			np.save(self.config['file_path.wiki_syn_file_npy'],word_syn)

		return word_syn

	def load_en_glyph_npy(self):
		if os.path.exists(self.config['file_path.wiki_glyph_file_npy']):
			glyph_edge = np.load(self.config['file_path.wiki_glyph_file_npy'],allow_pickle=True)
		else:
			num_sub, numword = \
						self.config['setting.num_sub'], self.config['setting.num_word']
			glyph_edge_list = [[] for i in range(num_sub + numword)]
			if self.split_type == 2:
				wiki_relation_file = self.config['file_path.wiki_wid2pieceid']
			with open(wiki_relation_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					assert len(info) == 3
					paper1 = int(info[0])+num_sub # word idx
					paper2 = int(info[1])
					type = int(info[2])

					if self.split_type == 2:
						glyph_edge_list[paper1].append((paper1,paper2,type))
						glyph_edge_list[paper2].append((paper2,paper1,type))
			glyph_edge = np.array(glyph_edge_list,dtype=object)
			np.save(self.config['file_path.wiki_glyph_file_npy'],glyph_edge)

		return glyph_edge

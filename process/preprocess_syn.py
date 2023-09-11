from nltk.corpus import wordnet as wn
from tqdm import tqdm
import argparse
import pyhocon

parser = argparse.ArgumentParser(description='The args of preprocessing background embeddings')
parser.add_argument('--config', type=str, default='../grm/experiments_en.conf')

args = parser.parse_args()

def get_word_list(config):
    wiki_word2id_r = open(config['file_path.wiki_word2id'],"r",encoding='utf-8')
    word_map = {}
    for line in wiki_word2id_r:
        lines = line.strip().split()
        id = lines[0]
        word = lines[1]
        word_map[word] = id
    return word_map

def get_syn_relation(config, word_map):
    syn_w = open(config['file_path.wiki_word_syn_file'],"w",encoding='utf-8')
    syn_set = {}
    for word in word_map.keys():
        w_syns = list(set([str(lemma.name()) for s in wn.synsets(word) for lemma in s.lemmas()]))
        w_syns_filtered = [s for s in w_syns if s != word and s in word_map]
        if len(w_syns_filtered) > 0:
            wid_syns_filtered = [word_map[s] for s in w_syns_filtered]
            syn_set[word_map[word]] = wid_syns_filtered

        if len(syn_set) >= 10000:
            for w, s_list in syn_set.items():
                for s in s_list:
                    syn_w.write(w+" "+s+" "+"1\n")
            # samples = [w+'\t'+(' '.join(s)) for w, s in syn_set.items()]
            # print(samples[0])
            # syn_w.write('\n'.join(samples))
            # syn_w.write('\n')
            syn_set = {}
        
    # samples = [w+'\t'+(' '.join(s)) for w, s in syn_set.items()]
    # print(samples[0])
    # syn_w.write('\n'.join(samples))
    # syn_w.write('\n')
    for w, s_list in syn_set.items():
        for s in s_list:
            syn_w.write(w+" "+s+" "+"1\n")


if __name__ == '__main__':
    # load config file
    config = pyhocon.ConfigFactory.parse_file(args.config)
    word_map = get_word_list(config)
    get_syn_relation(config, word_map)
    print("DONE!!!")

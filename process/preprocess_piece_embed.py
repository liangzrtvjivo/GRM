import pyhocon
import argparse

parser = argparse.ArgumentParser(description='The args of preprocessing background embeddings')
parser.add_argument('--config', type=str, default='../grm/experiments_en.conf')
parser.add_argument('--piece_emb_file', type=str, required=True)
args = parser.parse_args()

# load config file
config = pyhocon.ConfigFactory.parse_file(args.config)

piece_emb_file_r = open(args.piece_emb_file,"r",encoding='utf-8')
wiki_piece2id_r = open(config['file_path.wiki_piece2id'],"r",encoding='utf-8')
wiki_pieceid2feature_w = open(config['file_path.wiki_pieceid2feature'],"w",encoding='utf-8')

piece2feature_map = {}

dim = 0
index = -1
for line in piece_emb_file_r:
    if index<0:
        dim = int(line.strip().split(" ")[1])
        index+=1
        continue
    lines = line.strip().split(" ",1)
    piece = lines[0]
    feature = lines[1]
    piece2feature_map[piece] = feature
    index+=1

for line in wiki_piece2id_r:
    lines = line.strip().split()
    id = lines[0]
    piece = lines[1]
    if piece in piece2feature_map:
        feature = piece2feature_map[piece]
    else:
        feature = ' '.join(['0']*dim)
    wiki_pieceid2feature_w.write(f"{id} {feature}\n")
    

print("DONE!!!")
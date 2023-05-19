import argparse
import pyhocon
import torch
import os
import logging
from models import *
from dataCenter import DataCenter
from loader import OOVLoader

parser = argparse.ArgumentParser(description='infer args of GRM')
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--oov_file', type=str, default='../Wiki_title/zh_OOV_convert.txt')
parser.add_argument('--config', type=str, default='./experiments.conf')
parser.add_argument('--cuda', type=int, default=-1,
                    help='Avaiable GPU ID')
parser.add_argument('--batch_size', help='the number of samples in one batch', type=int, default=256)
parser.add_argument('--model_file', type=str, default='GRM_epoch_1.pt')
parser.add_argument('--ctrlname', type=str, default='2022-07-13-15_52_24')
parser.add_argument('--output_file', type=str, default='_wiki_title')
args = parser.parse_args()

model_file = './result/resultfile_' + args.ctrlname + '/' + args.model_file
logging_file = './result/resultfile_' + args.ctrlname  + '/logging_infer.txt'
output = './result/resultfile_' + args.ctrlname  + '/oov_result'+ args.output_file +'.txt'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S', filename = logging_file, filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)

device = torch.device("cuda:%d" % args.cuda if args.cuda != -1 else "cpu")
if torch.cuda.is_available():
    if args.cuda == -1:
        logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
        device_id = args.cuda
        logger.info('using device: ' + str(device_id) + ' -- '+ str(torch.cuda.get_device_name(device_id)))
device_name = "gpu" if args.cuda != -1 else "cpu"
logger.info('DEVICE:' + device_name)
outfile = open(output,"w", encoding='utf-8')

def set_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda != -1 and not torch.cuda.is_available():  # cuda is not available
        args.cuda = -1
    torch.manual_seed(args.seed)
    if args.cuda != -1:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

if __name__ == '__main__':
    logger.info('Data Processing Start!')
    set_seed(args)
    # load config file
    config = pyhocon.ConfigFactory.parse_file(args.config)
    oov_file = args.oov_file
    # load data
    dataCenter = DataCenter(config)
    dataCenter.load_dataSet()
    # get direct neigh
    glyph_edge,char_token_list,piece_token_list,glyph_add_map,oov_word_map = process_oov_file_by_piece(oov_file,config)

    nfeat = getattr(dataCenter, 'feats').shape[1]
    outfile.write(f'{len(oov_word_map)} {nfeat}\n')
    # process and split data
    data_loader = OOVLoader(args, config, dataCenter, glyph_edge,char_token_list,piece_token_list,glyph_add_map)
    infer_iterator = data_loader()
    logger.info('Data Processing done!')

    # load model
    embed_model = torch.load(model_file)

    with torch.no_grad():
        batch_index = 0
        for nodes_batch, raw_features, glyph_adj_batch, token_data in infer_iterator:
            pred_embs = embed_model.infer(raw_features, glyph_adj_batch, token_data)
            raw_index = 0
            for idx in nodes_batch:
                oov_word = oov_word_map[idx]
                oov_emb = pred_embs[raw_index].tolist()
                outfile.write(oov_word + ' ' + ' '.join([str(x) for x in list(oov_emb)]) + '\n')
                raw_index+=1

            batch_index+=1
            if batch_index%10==0:
                logger.info(f'Batch [{batch_index}/{len(infer_iterator)}] Done!')
        logger.info(f'Infer Done!')
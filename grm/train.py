# from ast import arg
import os
# import torch
import argparse
import pyhocon
import random
import time
import logging
from torch.optim import lr_scheduler
from loader import SimpleLoader
from loss import registry as loss_f
from embed_model import registry as model_f
from dataCenter import DataCenter
import torch.optim as optim
import gc
from utils import *

# time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
parser = argparse.ArgumentParser(description='train args of GRM')

parser.add_argument('--ctrlname', type=str)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--cuda', type=int, default=-1,
                    help='Avaiable GPU ID')
parser.add_argument('--config', type=str, default='./experiments.conf')
parser.add_argument('--batch_size', help='the number of samples in one batch', type=int, default=16)
parser.add_argument('--loss_type', help='ntx, align_uniform', type=str, default='ntx')
parser.add_argument('--shuffle', help='whether shuffle the samples', type=bool, default=True)
parser.add_argument('--learning_rate', help='learning rate for training', type=float, default=2e-3)
parser.add_argument('--gamma', help='decay rate', type=float, default=0.97)
parser.add_argument('--model_type', help='glyph_small_bn, glyph_sa_1_small_bn, glyph_sa_1_small_no_mask_bn, glyph_sa_1_small_no_decoder_bn', type=str, default='glyph_sa_1_small_bn')
parser.add_argument('--accumulation_steps', type=int, default=4)
parser.add_argument('--encoder_gnn_type', help='gcn, gat', type=str, default='gcn')
parser.add_argument('--probs', help='syn,sim,unchange', nargs='+', default=[0.1,0.7,0.2])
parser.add_argument('--isPosE', help='whether apply position embedding', type=bool, default=True)

args = parser.parse_args()

ctrlname = args.ctrlname
if ctrlname is None:
    ctrlname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
print(ctrlname)
if not os.path.exists('./result/resultfile_' + ctrlname):
    os.makedirs('./result/resultfile_' + ctrlname)
logging_file = './result/resultfile_' + ctrlname + '/logging.txt'
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

def calculate_loss(nodes_batch, pred_embs, glyph_data, device):
    pre_loss = None
    raw_features = getattr(dataCenter, 'feats')
    raw_glyph_adj_batch = glyph_data[3]

    probs = list(map(float,args.probs))
    batch_target_embs = torch.from_numpy(get_data_augment(nodes_batch,raw_glyph_adj_batch,config,dataCenter,raw_features,probs)).to(device)
    pre_loss = criterion(batch_target_embs, pred_embs, device)

    return pre_loss

def train_embed(epoch):
    epoch_loss = 0
    batch_index = 0
    for nodes_batch, features, glyph_data, token_data in train_iterator:
        pred_embs,other_loss = embed_model(features, glyph_data, token_data)
        if batch_index%50==0:
            print("pred_embs",pred_embs)
        if batch_index % 10 == 0:
            print_loss = epoch_loss/(batch_index+1)
            log_info = f'Batch [{batch_index}/{len(train_iterator)}] Done!loss = {print_loss}'
            logger.info(log_info)
        pre_loss = calculate_loss(nodes_batch, pred_embs, glyph_data, device)
        loss = pre_loss
        if other_loss != None:
            loss = loss + other_loss
        loss.backward()

        if (batch_index+1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        batch_index+=1
        epoch_loss = epoch_loss + loss.item()
    return epoch_loss

if __name__ == '__main__':
    set_seed(args)
    logger.info('Data Processing Start!')
    # load config file
    config = pyhocon.ConfigFactory.parse_file(args.config)

    # load data
    dataCenter = DataCenter(config)
    dataCenter.load_dataSet()

    # process and split data
    data_loader = SimpleLoader(args, config, dataCenter)
    train_iterator = data_loader()
    logger.info('Data Processing done!')
    
    # model and optimizer
    embed_model = model_f[args.model_type](config=config,
                        args=args,
                        device=device,
                        dataCenter=dataCenter)
    trainable_num = sum(p.numel() for p in embed_model.parameters() if p.requires_grad)
    print("trainable_num",trainable_num)
    optimizer = optim.Adam(embed_model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    criterion = loss_f[args.loss_type]()

    optimizer.zero_grad()

    # Train model
    logger.info("Training Begin!")
    t_total = time.time()
    for epoch in range(args.epochs):
        t = time.time()

        loss = train_embed(epoch)

        log =   'Epoch: {:d} '.format(epoch+1) + \
                'train: {:.4f} '.format(loss)
        logger.info(log)

        torch.save(embed_model, './result/resultfile_' + ctrlname +'/GRM_epoch_{}.pt'.format(epoch+1))
        logger.info(f"Epoch: {epoch+1} Finished! Model saved!")

    logger.info("Training Finished!")
    logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    del embed_model
    torch.cuda.empty_cache()
    gc.collect()
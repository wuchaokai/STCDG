import argparse
import torch
import os

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='GCN', help='model name | EvolveGCN | HTGN | WinGNN | SpikeNet | Roland')
parser.add_argument('--egcn_type', type=str, default='EGCNH', help='Type of EGCN: EGCNH or EGCNO')

parser.add_argument('--dataset', type=str, default='as733', help='dataset | as733 | enron10 | uci | dblp')

parser.add_argument('--time_steps', type=int, nargs='?', default=13, help="total time steps used for train, eval and test")


parser.add_argument('--nfeat', type=int, default=128, help='dim of input feature')
parser.add_argument('--out_feat', type=int, default=64, help='dim of output feature')
parser.add_argument('--log_path', type=str, default='./log', help='log path')

# 2.model
parser.add_argument('--dropout', type=float, default=0, help='dropout rate (1 - keep probability).')


# 3.experiment
parser.add_argument('--use_gpu', type=bool, default=True, help='use gru or not')
parser.add_argument('--device', type=str, default='cpu', help='training device')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')

parser.add_argument('--rewiring_type', type=str, default='base', help='rewiring_type')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size (# nodes)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight for L2 loss on basic models.')
parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
                        help='Weightage for negative samples')
parser.add_argument("--early_stop", type=int, default=20,
                        help="patient")
parser.add_argument('--limit_time',type=int,default=0)

parser.add_argument('--neg_sample_size', type=int, default=10, help='number of positive samples')
parser.add_argument('--window',type=int, default=1000)

# LSTM
parser.add_argument('--cov_num', type=int, default=1, help='layers of  gcn cov.')
parser.add_argument('--in_feature_list', type=list, default=[143], help='in feature of each layer.')
parser.add_argument('--gcn_drop', type=float, default=0.2, help='dropout of gcn.')

# EvolveGCN
parser.add_argument('--nhid', type=int, default=64, help='dim of hidden embedding')
parser.add_argument('--nout', type=int, default=16, help='dim of output embedding')
parser.add_argument('--sampling_times', type=int, default=1, help='negative sampling times')

parser.add_argument('--featureless', type=bool, nargs='?', default=True,
                    help='True if one-hot encoding.')
parser.add_argument('--inductive', type=bool, nargs='?', default=True,help='True if one-hot encoding.')


parser.add_argument('--compensate',action='store_true', default=False,help='True if compensating based model')
parser.add_argument('--without',type=str,default='None')
parser.add_argument('--type',type=str,default='ST')

parser.add_argument('--chooserate',type=int,default=1)
parser.add_argument('--threshold',type=float,default=100)
parser.add_argument('--neg_sample_path',type=str,default='neg_sample.pt')
parser.add_argument('--evaluation_data_filename',type=str,default='evaluation_data.npz')
parser.add_argument('--num_runs',type=int,default=10)


parser.add_argument('--sample_type',type=str,default='random')
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--select_rate',type=float,default=1.0)
parser.add_argument('--match_rate',type=float,default=1.0)

parser.add_argument('--re_w',type=float,default=1)
parser.add_argument('--trend_w',type=float,default=1)
parser.add_argument('--comb',type=str,default='concat')

parser.add_argument('--gen_sim', type=bool, default=False, help='whether generate the space similar matrix')
parser.add_argument('--sim_type',type=str,default='dtw', help='dtw/pagerank')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

# set the running device
if torch.cuda.is_available() and args.use_gpu:
    #print('using gpu:{} to train the model'.format(args.device_id))
    args.device_id = list(range(torch.cuda.device_count()))
    args.device = torch.device("cuda:{}".format(0))

else:
    args.device = torch.device("cpu")
    #print('using cpu to train the model')




if args.model == "EvolveGCN":
    args.nout = args.out_feat
    if args.dataset=='uci':
        args.learning_rate=0.005
    elif args.dataset=='as733':
        args.learning_rate = 0.001
    elif args.dataset=='enron10':
        args.learning_rate = 0.001
        #args.threshold=20

elif args.model == "HTGN":
    if args.dataset == 'enron10':
        args.learning_rate = 0.001


if args.model=='RTGCN':
    if args.dataset=='uci':
        args.re_w=0.1
        args.trend_w=1
    elif args.dataset=='dblp':
        args.re_w = 0   #0
        args.trend_w = 0.01 #0.01

    elif args.dataset=='enron10':
        args.re_w = 0.01
        args.trend_w = 1
    elif args.dataset=='as733':
        args.re_w = 0.01
        args.trend_w = 1

if args.model=='HTGN':
    if args.dataset=='dblp':
        args.re_w=0.1
        args.trend_w=1


if args.model=='SpikeNet':
    if args.dataset=='uci':
        args.re_w=0
        args.trend_w=0.1
        args.comb='mean'


    if args.dataset=='as733':
        args.re_w = 0.1
        args.trend_w = 0.1
        args.comb='mean'
        # args.re_w = 0.1
        # args.trend_w = 0.1
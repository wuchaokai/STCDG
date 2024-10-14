#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import logging
import os
from yacs.config import CfgNode as CN
import argparse
cfg = CN()

def set_cfg(cfg):
    r'''
    This function sets the default config value.

    :return: configuration use by the experiment.
    '''
    cfg.log_path = 'model_log/con_log'
    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #
    cfg.model = CN()

    # Link_prediction Edge_decoding
    cfg.model.edge_decoding = 'dot'

    # ------------------------------------------------------------------------ #
    # GNN options
    # ------------------------------------------------------------------------ #
    cfg.gnn = CN()
    # GNN skip connection
    cfg.gnn.skip_connection = 'affine'

    cfg.metric = CN()
    cfg.metric.mrr_method = 'max'


set_cfg(cfg)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='uci-msg', help='Dataset')
parser.add_argument('--egcn_type', type=str, default='EGCNH', help='Type of EGCN: EGCNH or EGCNO')
parser.add_argument('--cuda_device', type=int,
                    default=6, help='Cuda device no -1')

parser.add_argument('--seed', type=int, default=2023, help='split seed')

parser.add_argument('--repeat', type=int, default=10, help='number of repeat model')

parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train.')

parser.add_argument('--out_dim', type=int, default=64,
                    help='model output dimension.')

parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimizer type')

parser.add_argument('--lr', type=float, default=0.02,
                    help='initial learning rate.')

parser.add_argument('--maml_lr', type=float, default=0.008,
                    help='meta learning rate')

parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay (L2 loss on parameters).')

parser.add_argument('--drop_rate', type=float, default=0.16, help='drop meta loss')

parser.add_argument('--num_layers', type=int,
                    default=2, help='GNN layer num')

parser.add_argument('--num_hidden', type=int, default=256,
                    help='number of hidden units of MLP')
parser.add_argument('--compensate',action='store_true', default=False,help='True if compensating based model')
parser.add_argument('--without',type=str,default='space')
parser.add_argument('--model',type=str,default='space')
parser.add_argument('--window_num', type=float, default=8,
                    help='windows size')

parser.add_argument('--dropout', type=float, default=0.1,
                    help='GNN dropout')

parser.add_argument('--residual', type=bool, default=False,
                    help='skip connection')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
parser.add_argument('--beta', type=float, default=0.89,
                    help='The weight of adaptive learning rate component accumulation')
parser.add_argument('--select_rate',type=float,default=0.5)
parser.add_argument('--match_rate',type=float,default=0.5)
parser.add_argument('--limit_time',type=int,default=0)
parser.add_argument('--rewiring_type', type=str, default='base', help='rewiring_type')
parser.add_argument('--gen_sim', type=bool, default=False, help='whether generate the space similar matrix')
parser.add_argument('--sim_type',type=str,default='dtw', help='dtw/pagerank')
args = parser.parse_args()
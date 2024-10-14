import numpy as np
import torch
from dataloader import load_graphs,get_evaluation_data,csr_to_edge_index, include_isolated_nodes,load_graphs2
import networkx as nx
from torch.utils.data import DataLoader
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch.nn as nn
from model.EvolveGCN.EvolveGCN import EvolveGCN
import math
import copy
from model.RTGCN.RTGCN import RTGCN
from model.RTGCN.utils1 import *
from eval.eval_link_prediction import evaluate_classifier, calculate_mrr

from model.CompensateLayer import CompensateLayer

from utils.randomdataset import RandomDataset
from model.HTGN.HTGN import HTGN
from config.HTGNconfig import args as HTGN_args
from config.WinGNNconfig import args as WinGNN_args
from model.WinGNN import WinGNN


from model.Spikenet.SpikeNet import SpikeNet

import argparse
from config.config import args
import time
from model.HTGN.loss import ReconLoss,VGAEloss
import pandas as pd
from config.Spikenetconfig import args as Spike_args

import random
import dgl
from scipy.sparse import csr_matrix
import networkx as nx

from dataloader import evolve_edges,csr_to_adj_tensor
import os
from utils.calculate_sim2 import generateStructWeight as generateStructWeight2, generateSimMatrix_Pagerank,caculateTimeSim,generateTimeWeight,complete_graph_with_all_nodes
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
from torch_geometric.data import Data
warnings.filterwarnings('ignore')
from model.static import borf
from model.static import fosr
from model.static.curvature import sdrf
from model.static.GCN import GCN
import scipy.sparse as sp
from model.MergeLayer import MergeLayer
def generateSDRFGraph(data,shape):
    data=sdrf(data)
    edge_index=data.edge_index
    matrix=np.zeros((shape,shape))
    for i in range(edge_index.shape[1]):
        row=int(edge_index[0][i])
        col=int(edge_index[1][i])
        matrix[row][col]=1
        matrix[col][row]=1
    return matrix
def sparse_mx_to_torch_sparse_tensor(adj):
    sparse_mx=sp.coo_matrix(adj)
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def generateFOSRGraph(data,shape):
    data,_,_=fosr.edge_rewire(data.edge_index.numpy(), num_iterations=50)
    edge_index = data
    matrix = np.zeros((shape, shape))
    for i in range(edge_index.shape[1]):
        row = int(edge_index[0][i])
        col = int(edge_index[1][i])
        matrix[row][col] = 1
        matrix[col][row] = 1
    return matrix
def generateBORFGraph(data,shape,dataname):
    edge_index, _ = borf.borf3(data,
                                                       loops=10,
                                                       remove_edges=True,
                                                       is_undirected=True,
                                                       batch_add=20,
                                                       batch_remove=20,
                                                       dataset_name=dataname,
                                                       graph_index=0)
    matrix = np.zeros((shape, shape))
    for i in range(edge_index.shape[1]):
        row = int(edge_index[0][i])
        col = int(edge_index[1][i])
        matrix[row][col] = 1
        matrix[col][row] = 1
    return matrix
def genGeoData(x,edge_index):
    edge_index=edge_index.tocoo()
    row=edge_index.row.tolist()
    col=edge_index.col.tolist()
    edge_index=torch.tensor([row,col])
    data=Data(x=x,edge_index=edge_index)
    return data

def save_to_csv(file_name, new_data):
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df = df.append(new_data, ignore_index=True)
    else:
        df = pd.DataFrame(new_data, index=[0])

    df.to_csv(file_name, index=False)

def get_loss(model,feed_dict,time_steps,neg_weight,embedding):
    node_1, node_2,node_1_negative, node_2_negative, graphs = feed_dict.values()
    sumloss=0
    lossfuction=BCEWithLogitsLoss()
    # run gnn
    for t in range(time_steps - 2):
        emb_t = embedding[t] # [N, F]
        source_node_emb = emb_t[node_1[t+1]]
        tart_node_pos_emb = emb_t[node_2[t+1]]
        source_node_neg_emb = emb_t[node_1_negative[t+1]]
        tart_node_neg_emb = emb_t[node_2_negative[t+1]]
        pos_score = torch.sum(source_node_emb * tart_node_pos_emb, dim=1)
        neg_score = torch.sum(source_node_neg_emb * tart_node_neg_emb, dim=1)
        pos_loss = lossfuction(pos_score, torch.ones_like(pos_score))
        neg_loss = lossfuction(neg_score, torch.zeros_like(neg_score))
        graphloss = pos_loss + neg_weight * neg_loss
        sumloss += graphloss
    return sumloss

# def get_loss2(model,feed_dict,time_steps,neg_weight,embedding):
#     node_1, node_2,node_1_negative, node_2_negative, graphs = feed_dict.values()
#     sumloss=0
#     lossfuction=BCEWithLogitsLoss()
#     # run gnn
#     for t in range(time_steps-2):
#         emb_t = embedding[t] # [N, F]
#         source_node_emb = emb_t[node_1[t]]
#         tart_node_pos_emb = emb_t[node_2[t]]
#         source_node_neg_emb = emb_t[node_1_negative[t]]
#         tart_node_neg_emb = emb_t[node_2_negative[t]]
#         pos_score = model[1](source_node_emb,tart_node_pos_emb)
#         neg_score = model[1](source_node_neg_emb,tart_node_neg_emb)
#         predicts = torch.cat([pos_score, neg_score], dim=0)
#         labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)
#
#         graphloss = lossfuction(predicts,labels)
#         sumloss += graphloss
#     return sumloss

def evaluation(model,feed_dict,emb_t,t):
    node_1, node_2, node_1_negative, node_2_negative, graphs = feed_dict.values()

    # run gnn
    source_node_emb = emb_t[node_1[t+1].cpu()]
    tart_node_pos_emb = emb_t[node_2[t+1].cpu()]
    source_node_neg_emb = emb_t[node_1_negative[t+1].cpu()]
    tart_node_neg_emb = emb_t[node_2_negative[t+1].cpu()]
    pos_score = torch.sum(source_node_emb * tart_node_pos_emb, dim=1)
    neg_score = torch.sum(source_node_neg_emb * tart_node_neg_emb, dim=1)
    predicts = torch.cat([pos_score, neg_score], dim=0).detach().cpu().numpy()
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0).detach().cpu().numpy()

    # test_roc_score = roc_auc_score(labels, predicts)
    # test_ap = average_precision_score(labels, predicts)
    # test_mrr = calculate_mrr(labels, predicts)
    # binary_pred = [1 if score >= 0.5 else 0 for score in predicts]
    # test_recall = recall_score(labels, binary_pred)
    # test_f1 = f1_score(labels, binary_pred)

    val_test_binary_pred = [1 if score >= 0.5 else 0 for score in predicts]
    val_test_result = [1 if a == b else 0 for a, b in zip(val_test_binary_pred, labels)]

    return test_roc_score, test_ap, test_mrr, test_recall, test_f1,val_test_result



def get_loss_each_timestep(feed_dict,emb_t,t):
    node_1, node_2, node_1_negative, node_2_negative, graphs = feed_dict.values()
    lossfuction = BCEWithLogitsLoss()
    # run gnn
    source_node_emb = emb_t[node_1[t+1]]
    tart_node_pos_emb = emb_t[node_2[t+1]]
    source_node_neg_emb = emb_t[node_1_negative[t+1]]
    tart_node_neg_emb = emb_t[node_2_negative[t+1]]
    pos_score = torch.sum(source_node_emb * tart_node_pos_emb, dim=1)
    neg_score = torch.sum(source_node_neg_emb * tart_node_neg_emb, dim=1)
    pos_loss = lossfuction(pos_score, torch.ones_like(pos_score))
    neg_loss = lossfuction(neg_score, torch.zeros_like(neg_score))
    graphloss = pos_loss + neg_loss*args.neg_weight

    return graphloss
def to_device(batch, device):
    feed_dict = copy.deepcopy(batch)
    node_1, node_2, node_1_negative,node_2_negative, graphs = feed_dict.values()
    # to device
    feed_dict["node_1"] = [x.to(device) for x in node_1]
    feed_dict["node_2"] = [x.to(device) for x in node_2]
    feed_dict["node_1_neg"] = [x.to(device) for x in node_1_negative]
    feed_dict["node_2_neg"] = [x.to(device) for x in node_2_negative]
    feed_dict["graphs"] = [g.to(device) for g in graphs]

    return feed_dict
def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

if __name__ == '__main__':

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # load data
    if args.dataset in ['uci']:
        graphs, adjs, feats = load_graphs(args.dataset)
    elif args.dataset in ['as733','dblp','enron10']:
        graphs, adjs, feats = load_graphs2(args.dataset)

    unique_edges = set()
    for graph in graphs:
        for edge in graph.edges():
            unique_edges.add(tuple(sorted(edge)))


    total_unique_edges = len(unique_edges)
    print('节点数：', feats[-1].shape[0])
    print("总的唯一边数:", total_unique_edges)

    rewiring_type=args.rewiring_type
    rewiring_adj=[]
    if rewiring_type == 'sdrf':
        print('rewiring type is sdrf')
        data = genGeoData(feats[-1], adjs[-3])
        rewiring_adj.append(generateSDRFGraph(data, feats[-1].shape[0]))
        data = genGeoData(feats[-1], adjs[-2])
        rewiring_adj.append(generateSDRFGraph(data, feats[-1].shape[0]))
    elif rewiring_type == 'fosr':
        print('rewiring type is fosr')
        data = genGeoData(feats[-1], adjs[-3])
        rewiring_adj.append(generateFOSRGraph(data, feats[-1].shape[0]))
        data = genGeoData(feats[-1], adjs[-2])
        rewiring_adj.append(generateFOSRGraph(data, feats[-1].shape[0]))
    elif rewiring_type == 'borf':
        print('rewiring type is borf')
        data = genGeoData(feats[-1], adjs[-3])
        rewiring_adj.append(generateBORFGraph(data, feats[-1].shape[0], args.dataset))
        data = genGeoData(feats[-1], adjs[-2])
        rewiring_adj.append(generateBORFGraph(data, feats[-1].shape[0], args.dataset))

    if args.dataset=='as733':
        args.early_stop=10

    if args.limit_time>1:
        if args.limit_time>len(adjs):
            exit()
        limit_time = len(adjs)-args.limit_time
        graphs,adjs,feats=graphs[limit_time:],adjs[limit_time:],feats[limit_time:]
        print(limit_time)
    args.time_steps = len(adjs)
    node_num = feats[0].shape[0]

    if args.gen_sim:


        print('start to compute weight!')
        #caculateTimeSim(adjs, node_num, path='data/{}/'.format(args.dataset))
        dir = 'data/' + args.dataset
        save_dir = dir + '/genG'

        for time in range(args.time_steps):
            g=graphs[time]
            # chooseNodeFromCommunity(g, dir + '/subgraph/' + str(time) + '/', Max_rate=1)
            # node_num = feats.shape[0]
            # generateSimMatrix(g, dir + '/subgraph/' + str(time) + '/', node_num, chooserate=1)
            path='data/{}/subgraph2/{}/'.format(args.dataset,str(time))
            #generateSimMatrix2(g,path,node_num,select_rate=1,match_rate=1)
            generateSimMatrix_Pagerank(g,path,node_num,select_rate=1,match_rate=1)

        exit()







    if args.compensate:
        dir='data/{}/subgraph2/'.format(args.dataset)
        weightList=generateStructWeight2(dir,args,graphs)
        weightList=[torch.from_numpy(weightList[i]).float().to(args.device) for i in range(args.time_steps)]

        path='data/{}/'.format(args.dataset)
        TimeWeight=generateTimeWeight(path,args)
        TimeWeight=torch.from_numpy(TimeWeight).float().to(args.device)
    # if args.featureless == True:
    #     feats = [scipy.sparse.identity(adjs[args.time_steps - 1].shape[0]).tocsr() for _ in adjs]

    assert args.time_steps <= len(adjs), "Time steps is illegal"

    #context_pairs_train = get_context_pairs(graphs, adjs)

    # Load evaluation data for link prediction.
    train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
    test_edges_pos, test_edges_neg = get_evaluation_data(graphs,args)
    # print("No. Train: Pos={}, Neg={} \nNo. Val: Pos={}, Neg={} \nNo. Test: Pos={}, Neg={}".format(
    #     len(train_edges_pos), len(train_edges_neg), len(val_edges_pos), len(val_edges_neg),
    #     len(test_edges_pos), len(test_edges_neg)))
    # label_data = {}
    # val_edge = np.concatenate((val_edges_pos, val_edges_neg))
    # test_edge = np.concatenate((test_edges_pos, test_edges_neg))
    # edges = np.concatenate((val_edge, test_edge))
    #
    # val_label = np.concatenate((np.array([1]*len(val_edges_pos)), np.array([0]*len(val_edges_neg))))
    # test_label = np.concatenate((np.array([1]*len(test_edges_pos)), np.array([0]*len(test_edges_neg))))
    # val_test_label = np.concatenate((val_label, test_label))
    # for edge, result in zip(edges, val_test_label):
    #     label_data[str(edge)] = result
    # save_to_csv('data/{}/ST_compensation.csv'.format(args.dataset), label_data)
    # exit()
    # if args.inductive:
    #     new_G = inductive_graph(graphs[args.time_steps - 2], graphs[args.time_steps - 1])
    #     graphs[args.time_steps - 1] = new_G
    #     adjs[args.time_steps - 1] = nx.adjacency_matrix(new_G)

    device=args.device
    #dataset = MyDataset(args, graphs, feats, adjs, context_pairs_train)
    dataset=RandomDataset(args,graphs,feats,adjs)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0,
                            collate_fn=RandomDataset.collate_fn)

    auc_list,ap_list,recall_list,f1_list=[],[],[],[]
    model=None
    args.nfeat=feats[0].shape[1]
    if args.model=='EvolveGCN':
        model=EvolveGCN(nfeat=feats[0].shape[1],nhid=args.nhid,out_feat=args.nout,egcn_type=args.egcn_type,args=args,node_num=node_num).to(device)
        model=nn.Sequential(model)
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.model=='HTGN':

        HTGN_args.num_nodes=node_num
        HTGN_args.nfeat = feats[0].shape[1]
        model=HTGN(HTGN_args).to(device)
        model=nn.Sequential(model)
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        args.type='S'
    elif args.model=='WinGNN':

        model=WinGNN.Model(in_features=feats[0].shape[1],out_features=args.nout,hidden_dim=WinGNN_args.num_hidden,dropout=WinGNN_args.dropout,num_layers=WinGNN_args.num_layers).to(device)
        model=nn.Sequential(model)
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=WinGNN_args.weight_decay)
    elif args.model=='SpikeNet':


        new_adjs = [include_isolated_nodes(adj,node_num) for adj in adjs]
        evolve_adjs=[evolve_edges(adjs, node_num)[t] for t in range(len(adjs))]
        #now_adjs=accumulate_edges(new_adjs,node_num)
        model = SpikeNet(feats[0].shape[1], args.nout, alpha=Spike_args.alpha,
                         dropout=Spike_args.dropout, sampler=Spike_args.sampler, p=Spike_args.p,
                         aggr=Spike_args.aggr, concat=Spike_args.concat, sizes=Spike_args.sizes, surrogate=Spike_args.surrogate,
                         hids=Spike_args.hids, act=Spike_args.neuron, bias=True,adj=new_adjs,adj_evolve=evolve_adjs).to(device)
        model = nn.Sequential(model)
        opt=torch.optim.AdamW(model.parameters(), lr=Spike_args.lr)
    elif args.model=='Roland':
        model=Roland(feats[0].shape[1],args.nout)
        model=nn.Sequential(model).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=Roland_args.optim.base_lr,
                               weight_decay=Roland_args.optim.weight_decay)

    elif args.model=='DGTL':
        GAT_heads = [2, 1]
        model=DGTL_model(feats[0].shape[1],args.nout,[64,16],None,args.time_steps-1,'GCN',[0.2, 0],GAT_heads)
        opt = torch.optim.Adam(model.parameters(),lr=0.002,weight_decay=0)

        decayW = 1
        decay_length = args.time_steps-2 - 1
        decay_vec = torch.ones(1, decay_length)
        for i in range(decay_length):
            decay_vec[0][i] = math.exp(-decayW * (decay_length - i - 1))
        # for i in range(time_vector.size(0) - 1):
        #     if i == 0:
        #         decay_tensor = torch.cat((decay_vec, decay_vec), 0)
        #     else:
        #         decay_tensor = torch.cat((decay_tensor, decay_vec), 0)
        # decay_tensor = decay_tensor.to(device)


        for i in range(args.time_steps):
            adj = torch.tensor(nx.adjacency_matrix(complete_graph_with_all_nodes(graphs[i],node_num)).todense()).unsqueeze(2).to(device)
            feat=torch.tensor(feats[i]).unsqueeze(2).to(device)
            if i==0:
                features=feat
                adjs_new=adj
            else:
                features=torch.cat((features,feat),2)
                adjs_new=torch.cat((adjs_new,adj),2)
    elif args.model=='RTGCN':
        train_role_graph={}
        for index,graph in enumerate(graphs):
            train_role_graph[index]=wl_role_discovery(graph,4)
            # Process structural role data into a list
        list_loss_role = []
        for h in train_role_graph:
            list_g = []
            for g in train_role_graph[h]:
                list_g.append(torch.tensor(list(map(int, train_role_graph[h][g]))).to(device))
            list_loss_role.append(list_g)

            # Contruct role hypergraph
            train_hypergraph = []
            cross_role_hyper = []
            cross_role_laplacian = []  # leanth=time_step-1
            Role_set, Cross_role_Set = hypergraph_role_set(train_role_graph, args.time_steps)
            for i in range(args.time_steps):
                Role_hyper, H = gen_attribute_hg(node_num, train_role_graph[i], Role_set, X=None)
                train_hypergraph += [scipy_sparse_mat_to_torch_sparse_tensor(Role_hyper.laplacian()).to(device)]
                if i > 0:
                    previous_role_hypergraph, Cross_role_hypergraph = cross_role_hypergraphn_nodes(node_num, H,
                                                                                                   train_role_graph[
                                                                                                       i - 1],
                                                                                                   train_role_graph[i],
                                                                                                   Role_set, w=-11,
                                                                                                   delta_t=1,
                                                                                                   X=None)  #
                    cross_role_hyper += [scipy_sparse_mat_to_torch_sparse_tensor(Cross_role_hypergraph).to(device)]
                    cross_role_laplacian += [
                        scipy_sparse_mat_to_torch_sparse_tensor(previous_role_hypergraph.laplacian()).to(device)]
        Data_dblp = []
        attribute_matrix=np.transpose(np.array(feats),(1,0,2))
        Data_dblp.append(
            [torch.from_numpy(attribute_matrix[:, i, :]).float().to(device) for i in range(args.time_steps)])
        Data_dblp.append([csr_to_adj_tensor(adjs[j],node_num).to(device) for j in range(args.time_steps)])
        new_adjs = [include_isolated_nodes(adj, node_num) for adj in adjs]
        Data_dblp.append([scipy_sparse_mat_to_torch_sparse_tensor(new_adjs[w]).to(device) for w in range(args.time_steps)])
        model=RTGCN(act=nn.ELU(),
                          n_node=node_num,
                          input_dim=node_num,
                          output_dim=args.nout,
                          hidden_dim=args.nhid,
                          time_step=args.time_steps,
                          neg_weight=10,
                          loss_weight=1,
                          attn_drop=0.0,
                          residual=False,
                          role_num=len(Role_set),
                          cross_role_num=len(Role_set))
        model = nn.Sequential(model).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.model=='GCN':
        model=GCN(node_num,args.nhid,args.nout,0.5).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.compensate:
        args.node_num=node_num
        evolve_adjs = [csr_to_adj_tensor(evolve_edges(adjs, node_num)[t], node_num).to(args.device) for t in
                       range(len(adjs))]
        #TimeWeight=list(torch.unbind(TimeWeight, dim=0))
        comlayer=CompensateLayer(args,weightList,evolve_adjs,TimeWeight)
        # modules=list(model.children())
        # modules.append(comlayer)
        # model=nn.Sequential(*modules).to(device)
        model=model.append(comlayer).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        #model=nn.Sequential(model,comlayer).to(device)
    # LinkPredictLayer=MergeLayer(args.nout,args.nout,args.nout,1).to(device)
    # model.append(LinkPredictLayer)


    best_epoch_val = 0
    best_auc_test = 0
    best_ap = 0
    best_recall = 0
    best_f1 = 0
    patient = 0

    #training
    #if args.model=='EvolveGCN':
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for idx, feed_dict in enumerate(dataloader):
            feed_dict = to_device(feed_dict, device)
            opt.zero_grad()
            if args.model == 'EvolveGCN':
                embedding =(model(feed_dict['graphs']))
                loss = get_loss(model,feed_dict,args.time_steps,args.neg_weight,embedding)
                loss.backward()
                opt.step()
                epoch_losses.append(loss.item())
            elif args.model=='HTGN' and args.compensate==False:
                loss = ReconLoss(HTGN_args) if args.model not in ['DynVAE', 'VGRNN', 'HVGRNN'] else VGAEloss(args)
                edge_index_list=[feed_dict['graphs'][t].edge_index for t in range(args.time_steps)]
                model[0].init_hiddens()
                for t in range(args.time_steps - 2):
                    edge_index=edge_index_list[t]
                    opt.zero_grad()
                    x = dataset.features
                    z = model[0](edge_index.to(args.device), torch.from_numpy(x[t]).float().to(args.device))
                    if HTGN_args.use_htc == 0:
                        epoch_loss = loss(z, edge_index.to(args.device))
                    else:
                        epoch_loss = loss(z, edge_index.to(args.device)) + model[0].htc(z)
                    epoch_loss.backward()
                    opt.step()
                    epoch_losses.append(epoch_loss.item())
                    model[0].update_hiddens_all_with(z)
            elif args.model=='HTGN' and args.compensate:
                loss = ReconLoss(HTGN_args) if args.model not in ['DynVAE', 'VGRNN', 'HVGRNN'] else VGAEloss(args)
                edge_index_list=[feed_dict['graphs'][t].edge_index for t in range(args.time_steps)]
                model[0].init_hiddens()
                z_list=[]
                torch.autograd.set_detect_anomaly(True)
                epoch_loss=0
                for t in range(args.time_steps - 2):
                    edge_index=edge_index_list[t]
                    opt.zero_grad()
                    x = dataset.features
                    z = model[0](edge_index.to(args.device),torch.from_numpy(x[t]).float().to(args.device))
                    z_list.append(z)
                    z=model[1](z_list)[-1]
                    if HTGN_args.use_htc == 0:
                        epoch_loss =loss(z, edge_index.to(args.device))
                    else:
                        sigmoid=nn.Sigmoid()
                        z=sigmoid(z)
                        epoch_loss =epoch_loss+ loss(z, edge_index.to(args.device))
                        #model[0].update_hiddens_all_with(z)
                epoch_loss.backward()
                opt.step()
                epoch_losses.append(epoch_loss.item())
                #model[0].update_hiddens_all_with(z)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            elif args.model=='WinGNN':
                i=0
                model.train()
                graph=[dgl.graph((feed_dict['graphs'][t].edge_index[0],feed_dict['graphs'][t].edge_index[1]),num_nodes=feed_dict['graphs'][-1].x.shape[1]) for t in range(args.time_steps)]
                fast_weights = list(map(lambda p: p[0], zip(model[0].parameters())))
                S_dw = [0] * len(fast_weights)

                while i < (args.time_steps-WinGNN_args.window_num):
                    # if i != 0:
                    #     i = random.randint(i, i + WinGNN_args.window_num)
                    if i >= (args.time_steps - WinGNN_args.window_num-1):
                        break
                    losses = torch.tensor(0.0).to(device)
                    count = 0
                    emb_list=[]
                    for t in range(i,i+WinGNN_args.window_num):
                        graph[t].to(device)
                        feed_dict['graphs'][t].x.to(device)
                        pred=model[0](graph[t],feed_dict['graphs'][t].x,fast_weights)
                        emb_list.append(pred)
                        loss=get_loss_each_timestep(feed_dict,pred,t)
                        beta = WinGNN_args.beta
                        grad=torch.autograd.grad(loss, fast_weights,retain_graph=True, allow_unused=True)
                        S_dw = list(map(lambda p: beta * p[1] + (1 - beta) * p[0] * p[0], zip(grad, S_dw)))
                        fast_weights = list(
                            map(lambda p: p[1] - WinGNN_args.maml_lr / (torch.sqrt(p[2]) + 1e-8) * p[0],
                                zip(grad, fast_weights, S_dw)))
                        droprate = torch.FloatTensor(np.ones(shape=(1)) * WinGNN_args.drop_rate)
                        masks = torch.bernoulli(1. - droprate).unsqueeze(1)
                        if masks[0][0]:
                            losses = losses +loss
                            count += 1
                    if losses:
                        if args.compensate:
                            emb_list=model[1](emb_list)
                            losses=get_loss(model, feed_dict, len(emb_list), args.neg_weight, emb_list)/len(emb_list)+losses / count
                        else:
                            losses = losses / count
                        opt.zero_grad()
                        losses.backward()
                        opt.step()
                        epoch_losses.append(losses.cpu().detach().numpy())
                    i = i + 1

            elif args.model=='SpikeNet':
                model.train()
                opt.zero_grad()
                #nodes=torch.tensor(list(range(0,feed_dict['graphs'][0].x.shape[0])),dtype=torch.long)
                nodes=torch.arange(0,node_num)
                if args.compensate:
                    emb_list=[model[0](nodes,feed_dict['graphs'][:t+1]) for t in range(args.time_steps-1)]
                    emb_list=model[1](emb_list)
                    loss=get_loss(model,feed_dict,args.time_steps,1,emb_list)
                else:
                    emb = model[0](nodes, feed_dict['graphs'][:-2])
                    loss=get_loss_each_timestep(feed_dict,emb,args.time_steps-3)
                #loss=get_loss(model,feed_dict,args.time_steps,1,emb_list)

                loss.backward()
                opt.step()
                epoch_losses.append(loss.item())

            elif args.model=='DGTL':
                model.train()
                opt.zero_grad()
                if args.compensate:
                    pass
                else:
                    _,emb=model((features,adjs_new),feed_dict['graphs'][t].edge_index)
                    print(emb)
            elif args.model=='RTGCN':
                model.train()
                opt.zero_grad()
                if args.compensate:
                    emb_list=model[0].forward(Data_dblp, train_hypergraph, cross_role_hyper,cross_role_laplacian)
                    emb_list=model[1](emb_list)
                    loss = get_loss(model, feed_dict, args.time_steps, 1, emb_list)
                else:
                    loss=model[0].get_loss(feed_dict,Data_dblp,train_hypergraph,cross_role_hyper,cross_role_laplacian,list_loss_role)
                loss.backward()
                opt.step()
            elif args.model=='GCN':
                model.train()
                opt.zero_grad()
                emb=model(torch.from_numpy(feats[-1]).float().to(args.device),sparse_mx_to_torch_sparse_tensor(rewiring_adj[0]).to(args.device))
                loss=get_loss_each_timestep(feed_dict,emb,args.time_steps-3)
                loss.backward()
                opt.step()




        model.eval()
        if args.model == 'EvolveGCN':
            emb = model(feed_dict['graphs'])[-2].detach().cpu().numpy()
            #emb = (model(feed_dict['graphs']))
            #torch.save(emb, 'data/{}/spatialCompensate/embed.pt'.format(args.dataset))
        elif args.model=='HTGN' and args.compensate:
            emb = model[0](edge_index_list[-2],torch.from_numpy(dataset.features[-2]).float().to(args.device))
            emb_list=[emb]*(len(edge_index_list)-2)
            emb=model[1](emb_list)[-1].detach().cpu().numpy()
        elif args.model=='HTGN':
            emb = model[0](edge_index_list[-2],
                        torch.from_numpy(dataset.features[-2]).float().to(args.device)).detach().cpu().numpy()
        elif args.model=='WinGNN':

            emb=model[0](graph[-2],feed_dict['graphs'][-2].x,fast_weights).detach().cpu().numpy()
            if args.compensate:
                emb_list=[model[0](graph[i], feed_dict['graphs'][-2].x, fast_weights) for i in range(args.time_steps-1)]
                emb=model[1](emb_list)[-1].detach().cpu().numpy()

        elif args.model=='SpikeNet':
            emb = model[0](nodes,feed_dict["graphs"][:-1]).detach().cpu().numpy()
            if args.compensate:
                emb=emb_list[-1].detach().cpu().numpy()
        #emb=model[1](emb,weightList)[-2].detach().cpu().numpy()
        elif args.model=='RTGCN':
            emb_list=model[0].forward(Data_dblp, train_hypergraph, cross_role_hyper,cross_role_laplacian)
            if args.compensate:
                emb=model[1](emb_list)[-2].detach().cpu().numpy()
            else:
                emb=emb_list[-2].detach().cpu().numpy()
        elif args.model=='GCN':
            emb=model(torch.from_numpy(feats[-1]).float().to(args.device),sparse_mx_to_torch_sparse_tensor(rewiring_adj[1]).to(args.device)).detach().cpu().numpy()
        val_auc, test_roc_score, test_ap, test_mrr,test_recall,test_f1,val_test_results,val_test_pred = evaluate_classifier(train_edges_pos,
                                                              train_edges_neg,
                                                              val_edges_pos,
                                                              val_edges_neg,
                                                              test_edges_pos,
                                                              test_edges_neg,
                                                              emb,
                                                              emb)
        # val_auc, val_ap, _,_,_ = evaluation(model[1], feed_dict, emb[-2],
        #                                                                      args.time_steps - 2)
        #test_roc_score, test_ap, test_mrr, test_recall, test_f1,val_test_results = evaluation(model[1],feed_dict,embedding[-2],args.time_steps-2)

        if val_auc > best_auc_test:
            best_epoch_val = val_auc
            best_auc_test=test_roc_score
            best_ap=test_ap
            bets_mrr=test_mrr
            best_recall=test_recall
            best_f1=test_f1
            best_result=val_test_results
            best_pred=val_test_pred
            best_emb=emb
            #os.makedirs("./model_checkpoints",exist_ok=True)
            #torch.save(model.state_dict(), "./model_checkpoints/model.pt")
            patient = 0
            
        else:
            patient += 1
            if patient > args.early_stop:
                break


        print("Epoch {:<3},  Loss = {:.3f}, Val AUC {:.4f} Test AUC {:.4f} Test AP {:.4f} Test MRR {:.4f} Test Recall {:.4f} Test F1 {:.4f}".format(epoch,
                                                                                   np.mean(epoch_losses),
                                                                                   val_auc,test_roc_score,
                                                                                   test_ap,test_mrr,test_recall,test_f1))
    print(
        "Best Test AUC {:.4f} Test AP {:.4f} Test MRR {:.4f} Test Recall {:.4f} Test F1 {:.4f}".format(
            best_auc_test,
            best_ap, bets_mrr,best_recall, best_f1))
    #print(best_pred)
    #torch.save(best_emb, 'data/{}/NoCompensate/embed.pt'.format(args.dataset, args.time_steps))
    if args.compensate:
        if args.model=='EvolveGCN':
            torch.save(best_emb, 'data/{}/spatialTimeCompensate/embed_{}.pt'.format(args.dataset, args.egcn_type))
        else:
            torch.save(best_emb, 'data/{}/spatialTimeCompensate/embed_{}.pt'.format(args.dataset,args.model))
    else:
        torch.save(best_emb, 'data/{}/NoCompensate/embed_{}.pt'.format(args.dataset, args.model))
    new_data = {
        "best_auc_test": best_auc_test,
        "best_ap": best_ap,
        'best_mrr':bets_mrr,
        "best_recall": best_recall,
        "best_f1": best_f1
    }

    save_to_csv('data/{}/results.csv'.format(args.dataset), new_data)
    #
    # new_data2={}
    # val_edge=np.concatenate((val_edges_pos, val_edges_neg))
    # test_edge=np.concatenate((test_edges_pos, test_edges_neg))
    # edges=np.concatenate((val_edge, test_edge))
    # for edge,result in zip(edges,best_result):
    #     new_data2[str(edge)]=result
    # save_to_csv('data/{}/ST_compensation.csv'.format(args.dataset), new_data2)
    #     auc_list.append(best_auc_test)
    #     ap_list.append(best_ap)
    #     recall_list.append(best_recall)
    #     f1_list.append(best_f1)
    #     reset_parameters(model)
    # print(
    #         f"Best Test AUC {np.mean(auc_list):.4f}± {np.std(auc_list, ddof=1):.4f}'   Test AP {np.mean(ap_list):.4f}± {np.std(ap_list, ddof=1):.4f}' Test Recall {np.mean(recall_list):.4f}± {np.std(recall_list, ddof=1):.4f}' Test F1 {np.mean(f1_list):.4f}± {np.std(f1_list, ddof=1):.4f}'")

        #f'± {np.std(auc_list, ddof=1):.4f}'
        # Test Best Model
        # model.load_state_dict(torch.load("./model_checkpoints/model.pt"))
        # model.eval()
        # emb = model(feed_dict["graphs"])[-2].detach().cpu().numpy()
        # val_auc, test_roc_score, test_ap, test_recall,test_f1 = evaluate_classifier(train_edges_pos,
        #                                                       train_edges_neg,
        #                                                       val_edges_pos,
        #                                                       val_edges_neg,
        #                                                       test_edges_pos,
        #                                                       test_edges_neg,
        #                                                       emb,
        #                                                       emb)
        #
        # print(
        #     "Load Test AUC {:.4f} Test AP {:.4f} Test Recall {:.4f} Test F1 {:.4f}".format(
        #         test_roc_score,
        #         test_roc_score, test_ap, test_f1))
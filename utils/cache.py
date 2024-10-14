from config.config import args
import scipy
import numpy as np
import torch
from dataloader import load_graphs,get_evaluation_data
import networkx as nx
from torch.utils.data import DataLoader
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch.nn as nn
from model.EvolveGCN.EvolveGCN import EvolveGCN
import os
import copy
from eval.eval_link_prediction import evaluate_classifier
from utils.calculate_sim import chooseNodeFromCommunity,generateSimMatrix,generateStructWeight
from model.CompensateLayer import CompensateLayer
from utils.randomdataset import RandomDataset
def inductive_graph(graph_former, graph_later):
    """Create the adj_train so that it includes nodes from (t+1)
       but only edges from t: this is for the purpose of inductive testing.

    Args:
        graph_former ([type]): [description]
        graph_later ([type]): [description]
    """
    newG = nx.MultiGraph()
    newG.add_nodes_from(graph_later.nodes(data=True))
    newG.add_edges_from(graph_former.edges(data=False))
    return newG

def get_loss(model,feed_dict,time_steps,neg_weight):
    node_1, node_2, node_2_negative, graphs = feed_dict.values()
    sumloss=0
    embedding=model(graphs)
    lossfuction=BCEWithLogitsLoss()
    # run gnn
    for t in range(time_steps - 1):
        emb_t = embedding[t] # [N, F]
        source_node_emb = emb_t[node_1[t]]
        tart_node_pos_emb = emb_t[node_2[t]]
        tart_node_neg_emb = emb_t[node_2_negative[t]]
        pos_score = torch.sum(source_node_emb * tart_node_pos_emb, dim=1)
        neg_score = -torch.sum(source_node_emb[:, None, :] * tart_node_neg_emb, dim=2).flatten()
        pos_loss = lossfuction(pos_score, torch.ones_like(pos_score))
        neg_loss = lossfuction(neg_score, torch.ones_like(neg_score))
        graphloss = pos_loss + neg_weight * neg_loss
        sumloss += graphloss
    return sumloss

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


if __name__ == '__main__':
    np.random.seed(2024)
    torch.manual_seed(2024)

    # load data
    graphs, adjs, feats = load_graphs(args.dataset)
    args.time_steps = len(adjs)
    if args.gen_sim:
        dir = 'data/' + args.dataset
        save_dir = dir + '/genG'

        for time in range(args.time_steps):
            g=graphs[time]
            chooseNodeFromCommunity(g, dir + '/subgraph/' + str(time) + '/', Max_rate=1)
            node_num = feats.shape[0]
            generateSimMatrix(g, dir + '/subgraph/' + str(time) + '/', node_num, chooserate=1)

    if args.compensate:
        dir='data/{}/subgraph/'.format(args.dataset)
        weightList=generateStructWeight(dir,args)
        weightList=[torch.from_numpy(weightList[i]).float().to(args.device) for i in range(args.time_steps)]


    if args.featureless == True:
        feats = [scipy.sparse.identity(adjs[args.time_steps - 1].shape[0]).tocsr() for _ in adjs]

    assert args.time_steps <= len(adjs), "Time steps is illegal"

    #context_pairs_train = get_context_pairs(graphs, adjs)

    # Load evaluation data for link prediction.
    train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
    test_edges_pos, test_edges_neg = get_evaluation_data(graphs)
    print("No. Train: Pos={}, Neg={} \nNo. Val: Pos={}, Neg={} \nNo. Test: Pos={}, Neg={}".format(
        len(train_edges_pos), len(train_edges_neg), len(val_edges_pos), len(val_edges_neg),
        len(test_edges_pos), len(test_edges_neg)))


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
                            num_workers=10,
                            collate_fn=RandomDataset.collate_fn)

    model=None
    if args.model=='EvolveGCN':
        model=EvolveGCN(nfeat=feats[0].shape[1],nhid=args.nhid,out_feat=args.nout,egcn_type=args.egcn_type,args=args,node_num=graphs[-1].number_of_nodes()).to(device)
    if args.compensate:
        comlayer=CompensateLayer(args.nout,args.nout,args,weightList)
        model=nn.Sequential(model,comlayer).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_epoch_val = 0
    patient = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        for idx, feed_dict in enumerate(dataloader):
            feed_dict = to_device(feed_dict, device)
            opt.zero_grad()
            loss = get_loss(model,feed_dict,args.time_steps,args.neg_weight)
            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())

        model.eval()
        emb = model(feed_dict["graphs"])[-2].detach().cpu().numpy()
        #emb=model[1](emb,weightList)[-2].detach().cpu().numpy()
        val_auc, test_roc_score, test_ap, test_recall,test_f1 = evaluate_classifier(train_edges_pos,
                                                              train_edges_neg,
                                                              val_edges_pos,
                                                              val_edges_neg,
                                                              test_edges_pos,
                                                              test_edges_neg,
                                                              emb,
                                                              emb)

        if val_auc > best_epoch_val:
            best_epoch_val = val_auc
            best_auc_test=test_roc_score
            best_ap=test_ap
            best_recall=test_recall
            best_f1=test_f1
            os.makedirs("./model_checkpoints",exist_ok=True)
            torch.save(model.state_dict(), "./model_checkpoints/model.pt")
            patient = 0
        else:
            patient += 1
            if patient > args.early_stop:
                break

        print("Epoch {:<3},  Loss = {:.3f}, Val AUC {:.4f} Test AUC {:.4f} Test AP {:.4f} Test Recall {:.4f} Test F1 {:.4f}".format(epoch,
                                                                                   np.mean(epoch_loss),
                                                                                   val_auc,test_roc_score,
                                                                                   test_roc_score,test_ap,test_f1))
    print(
        "Best Test AUC {:.4f} Test AP {:.4f} Test Recall {:.4f} Test F1 {:.4f}".format(
            best_auc_test,
            best_ap, best_recall, best_f1))
    # Test Best Model
    model.load_state_dict(torch.load("./model_checkpoints/model.pt"))
    model.eval()
    emb = model(feed_dict["graphs"])[-2].detach().cpu().numpy()
    val_auc, test_roc_score, test_ap, test_recall,test_f1 = evaluate_classifier(train_edges_pos,
                                                          train_edges_neg,
                                                          val_edges_pos,
                                                          val_edges_neg,
                                                          test_edges_pos,
                                                          test_edges_neg,
                                                          emb,
                                                          emb)

    print(
        "Load Test AUC {:.4f} Test AP {:.4f} Test Recall {:.4f} Test F1 {:.4f}".format(
            test_roc_score,
            test_roc_score, test_ap, test_f1))
import pandas as pd
from fastdtw import fastdtw
import dgl
import torch
from math import exp
from sklearn.manifold import TSNE
import networkx as nx
import sys
import os
import numpy as np
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import pandas
import re, sys, math, random, csv, types, networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import scipy.sparse as sp
#import igraph as ig
#from karateclub import LabelPropagation,EdMot,SCD
import copy
import json

#import umap
# import dgl
# import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from mpl_toolkits.mplot3d import Axes3D
#from fastdtw import fastdtw
#from Pagerank import pr
import numpy as np
from scipy.spatial.distance import cosine
import community as community_louvain
def spilt_community(g,path):
    # nx.draw(g)
    # plt.show()
    #model =LabelPropagation()
    #model=EdMot()
    os.makedirs(path,exist_ok=True)
    if os.path.exists(path+'communitys.json') and os.path.exists(path+'cluster.json'):
        cluster_membership=readjson(path+'cluster.json')
        communitys=readjson(path+'communitys.json')
        cluster_membership={int(key):value for key,value in cluster_membership.items()}
        communitys = {int(key): value for key, value in communitys.items()}
        #return cluster_membership,communitys
    cluster_membership=community_louvain.best_partition(g)

    # model=LabelPropagation()
    # model.fit(g)
    # cluster_membership = model.get_memberships()
    print(cluster_membership)
    cluster_membership = {int(key): value for key,value in cluster_membership.items()}
    communitys = defaultdict(list)
    for key, val in cluster_membership.items():
        communitys[val].append(key)
    communitys = dict(communitys)
    print(cluster_membership)
    print(communitys)
    print(len(communitys))
    for key in communitys.keys():
        print(len(communitys[key]))
    filename_communitys=path+'communitys.json'
    filename_cluster=path+'cluster.json'
    savejson(communitys,filename_communitys)
    savejson(cluster_membership, filename_cluster)
    return cluster_membership,communitys
    # print(cluster_membership)
def calculateDegree(g,cluster,communitys):
    graph = nx.to_dict_of_lists(g)
    Degrees = defaultdict(int)
    Outdegrees = defaultdict(int)
    for i,_ in graph.items():
        degree = len(graph[i])
        label = cluster[int(i)]
        indegree = 0
        outdegree = 0
        for neigh in graph[i]:
            if int(neigh) in communitys[label]:
                indegree += 1
        outdegree = degree - indegree
        Degrees[int(i)] = degree
        Outdegrees[int(i)] = outdegree
    return Degrees,Outdegrees
def calculate_S_list(g,Degrees,Outdegrees,hop=0):
    S_degree = defaultdict(int)
    S_outdegree = defaultdict(int)
    if hop == 0:
        S_degree = copy.deepcopy(Degrees)
        S_outdegree = copy.deepcopy(Outdegrees)
    else:
        graph = g
        A = nx.adjacency_matrix(graph) ** hop + sp.csr_matrix(np.eye(len(graph)))
        for i, j in zip(A.nonzero()[0], A.nonzero()[1]):
            S_degree[i] += Degrees[j]
            S_outdegree[i] += Outdegrees[j]
    return S_degree,S_outdegree
def calculate_S_Escaping(members, S_degree,S_outdegree):
    community_member_er = defaultdict(float)
    for member in members:
        if S_degree[member] == 0:
            community_member_er[member] = 0
        else:
            community_member_er[member] = S_outdegree[member] / S_degree[member]
    return community_member_er
def get_lowest_percent(dict_obj,rate):
    sorted_items = sorted(dict_obj.items(), key=lambda x: x[1])
    lowest_count = int(len(dict_obj) * rate)
    return dict(sorted_items[:lowest_count])
def chooseNodeFromCommunity(g,path, Max_rate=0.5, Min_rate=0.1, Max_hop=3,Min_hop=2,chooserate=0.5,datalen=0,community_num=2):
    cluster,communitys=spilt_community(g,path)
    #cluster,communitys=readjson(path)
    #cluster, communitys = cluster_communitys(datalen,path,community_num)
    choose_nodeset = defaultdict(list)
    Degrees, Outdegrees = calculateDegree(g, cluster, communitys)
    S_list=[]
    for hop in range(0,Max_hop):
        S_list.append(calculate_S_list(g,Degrees,Outdegrees,hop))
    if len(communitys)==1:
        choose_nodeset=communitys
    else:
        for label, member in communitys.items():
            if len(member)<3:
                continue
            Last_nodeset = {}
            Node_set = {}
            if Max_rate<1:
                for hop in range(1,Max_hop):
                    S_degree,S_outdegree=S_list[hop]
                    S_escape_rate = calculate_S_Escaping(member,S_degree,S_outdegree)

                    # Sort_Score=sorted(S_escape_rate.items(),key=lambda x:x[1],reverse=False)
                    #Node_set = get_lowest_percent(S_escape_rate, Max_rate)
                    Node_set = {key: S_escape_rate[key] for key in S_escape_rate.keys() if S_escape_rate[key]<=0}
                    if len(Node_set) <= int(len(member) * Min_rate):
                        Node_set = get_lowest_percent(S_escape_rate,Max_rate)
                        print(len(Node_set), 'case1', Node_set)
                        break
                    elif len(Node_set) <= max(1, int(len(member) * Max_rate)):
                        print(len(Node_set), 'case2', Node_set)
                        break
            else:
                Node_set={id:0 for id in member}

            if len(Node_set) > max(1, int(len(member) * Max_rate)):
                random_choose = random.sample(Node_set.keys(), max(1, int(len(member) * Max_rate)))
                Node_set = dict(filter(lambda x: x[0] in random_choose, Node_set.items()))
                print(len(Node_set),'case4')
            if len(Node_set) > 0:
                # print(len(member),Node_set)
                choose_nodeset[label].extend(list(Node_set.keys()))
    print(choose_nodeset)
    print(len(choose_nodeset))
    savejson(dict(choose_nodeset), path=path+'node' + str(Max_rate) + '.json')
def chooseNodeFromCommunity_yu(g,path, yuzhi=0,Max_rate=0.5, Min_rate=0.1, Max_hop=5,Min_hop=2,chooserate=0.5,datalen=0,community_num=2):
    cluster,communitys=spilt_community(g,path)
    #cluster, communitys = cluster_communitys(datalen,path,community_num)
    choose_nodeset = defaultdict(list)
    nodesum=0
    for label, member in communitys.items():
        if len(member)<3:
            continue
        Last_nodeset = {}
        Node_set = {}
        S_escape=list()
        for hop in range(2):
            S_escape_rate = calculate_S_Escaping(g,member, cluster,communitys,hop)

            # Sort_Score=sorted(S_escape_rate.items(),key=lambda x:x[1],reverse=False)
            #Node_set = get_lowest_percent(S_escape_rate, Max_rate)
            Node_set = {key: S_escape_rate[key] for key in S_escape_rate.keys() if S_escape_rate[key]<=yuzhi}
            S_escape.extend(S_escape_rate.values())
            # if len(Node_set) <= int(len(member) * Min_rate):
            #     Node_set = get_lowest_percent(S_escape_rate,Max_rate)
            #     print(len(Node_set), 'case1', Node_set)
            #     break
            # elif len(Node_set) <= max(1, int(len(member) * Max_rate)):
            #     print(len(Node_set), 'case2', Node_set)
            #     break
            # elif len(Node_set) <= hop:
            #     # random_choose = random.sample(Node_set.keys(), max(1, int(len(member) * Max_rate)))
            #     # Node_set = dict(filter(lambda x: x[0] in random_choose, Node_set.items()))
            #     Node_set = get_lowest_percent(S_escape_rate, Max_rate)
            #     print(len(Node_set), 'case3', Node_set)
            #     break
            # else:
            #     # Last_nodeset = copy.deepcopy(Node_set)
            #     # random_choose = random.sample(Node_set.keys(), max(1, int(len(member) * Max_rate)))
            #     # Last_nodeset = dict(filter(lambda x: x[0] in random_choose, Last_nodeset.items()))
            #     Last_nodeset=get_lowest_percent(S_escape_rate,Max_rate)
        # if len(Node_set) > max(1, int(len(member) * Max_rate)):
        #     random_choose = random.sample(Node_set.keys(), max(1, int(len(member) * Max_rate)))
        #     Node_set = dict(filter(lambda x: x[0] in random_choose, Node_set.items()))
        #     print(len(Node_set),'case4')
        if len(Node_set) > 0:
            # print(len(member),Node_set)
            choose_nodeset[label].extend(list(Node_set.keys()))
            nodesum+=len(Node_set)
    print(choose_nodeset)
    print(len(choose_nodeset))
    savejson(dict(choose_nodeset), path=path+'node' +str(yuzhi)+ '.json')

def savejson(data, path='node.json'):
    info_json = json.dumps(data, separators=(',', ': '), indent=4)
    f = open(path, 'w')
    f.write(info_json)


def readjson(path='node.json'):
    f = open(path, 'r')
    info_data = json.load(f)
    return info_data
def cluster_communitys(datalen,path,community_num):
    labels=[]
    for id in range(community_num):
        labels=labels+[id for i in range(datalen)]
    cluster = {i: j for i, j in zip(range(datalen*(community_num)), labels)}
    communitys = {}
    for num in range(community_num):
        communitys[num]=list(range(datalen*num,datalen*(num+1)))
    filename_communitys = path + 'communitys.json'
    savejson(communitys, filename_communitys)
    return cluster, communitys

def generateSimMatrix(g,path,node_num,chooserate=0.5,start=0,khop=3):
    filename_node = path + 'node' + str(chooserate) + '.json'
    filename_communitys = path + 'communitys.json'
    graph = nx.to_dict_of_lists(g)
    os.makedirs(path, exist_ok=True)
    Degrees = defaultdict(int)
    for i in graph.keys():
        degree = len(graph[i])
        Degrees[int(i)] = degree

    data = readjson(path=filename_node)
    data={int(key):value for key,value in data.items()}
    #print(data)
    #print(len(data))
    nodeList = []
    for key, values in data.items():
        nodeList.extend(values)
    print(path,chooserate,len(nodeList)/len(g.nodes()))
    communitys=readjson(filename_communitys)
    communitys={int(key):value for key,value in communitys.items()}
    ret = defaultdict(list)
    if start!=0:
        simMatrix=np.load(path+'simMatrixHop' + str(start-1)+'_'+str(chooserate) + '.npy')
        ret = readjson(path+'ret.json')
    else:
        simMatrix = np.array([[-1.0 for i in range(node_num)] for j in range(node_num)])
        #simMatrix=np.zeros([matrix_size,matrix_size])
    for hop in range(start, khop):
        if hop == 0:
            for label, nodeid in data.items():  # 选中的点（label：nodeid）
                ret[nodeid[0]] = list(set(nodeList).difference(set(communitys[label])))
                if len(communitys)==1:
                    ret[nodeid[0]]=list(set(nodeList).difference(set([label])))
                #ret[str(nodeid[0])] = random.sample(ret[str(nodeid[0])], int(len(ret[str(nodeid[0])]) ))
                for nodei in nodeid:
                    for nodej in ret[nodeid[0]]:
                        if Degrees[nodei]!=0 and Degrees[nodej]!=0:
                            simMatrix[nodei][nodej]=cacDistance(Degrees[nodei],Degrees[nodej])
                            print(simMatrix[nodei][nodej], nodei, nodej, path + str(hop))
        else:
            A = nx.adjacency_matrix(g) ** hop + sp.csr_matrix(np.eye(len(graph)))
            for label, nodeid in data.items():  # 选中的点（label：nodeid）
                for nodei in nodeid:
                    for nodej in ret[nodeid[0]]:
                        if Degrees[nodei]!=0 and Degrees[nodej]!=0:
                            simMatrix[nodei][nodej] = simMatrix[nodei][nodej] + caculateSim(A, nodei, nodej, Degrees)
                            simMatrix[nodej][nodei] = simMatrix[nodei][nodej]
                            print(simMatrix[nodei][nodej], nodei, nodej,path+str(hop))
    np.save(path+'simMatrixHop' + '_'+str(chooserate) +'.npy', simMatrix)
    print('save simMatrixHop'+ '_'+str(chooserate) + '.npy in '+path)
        #savejson(ret,path+'ret.json')
def generateSimMatrix_feature(g,path,features,chooserate=0.5):
    filename_node = path + 'node' + str(chooserate) + '.json'
    filename_communitys = path + 'communitys.json'
    graph = g
    matrix_size = len(graph)
    data = readjson(path=filename_node)
    #print(data)
    #print(len(data))
    nodeList = []
    for key, values in data.items():
        nodeList.extend(values)
    print(path,chooserate,len(nodeList)/len(g.nodes()))
    communitys=readjson(filename_communitys)
    ret = defaultdict(list)
    simMatrix = np.array([[-1.0 for i in range(matrix_size)] for j in range(matrix_size)])
    #simMatrix=np.zeros([matrix_size,matrix_size])

    for label, nodeid in data.items():  # 选中的点（label：nodeid）
        ret[str(nodeid[0])] = list(set(range(matrix_size)).difference(set(communitys[label])))
        #ret[str(nodeid[0])] = random.sample(ret[str(nodeid[0])], int(len(ret[str(nodeid[0])]) ))
        for nodei in nodeid:
            for nodej in ret[str(nodeid[0])]:
                simMatrix[nodei][nodej]=cosine(features[nodei],features[nodej])
    mkdir(path+'feature')
    np.save(path+'feature/'+'simMatrixHop' +'_'+str(chooserate) + '.npy', simMatrix)
    print('save simMatrixHop_feature'+ '_'+str(chooserate) + '.npy')
    savejson(ret,path+'feature/'+'ret.json')
    return graph
def generateSimMatrix_yuzhi(g,path,yuzhi=0.5,start=0,khop=6):
    filename_node = path + 'node' + str(yuzhi) + '.json'
    filename_communitys = path + 'communitys.json'
    graph = g
    Degrees = defaultdict(int)
    for i in range(len(graph)):
        degree = len(graph[i])
        Degrees[i] = degree
    matrix_size = len(graph)
    data = readjson(path=filename_node)
    #print(data)
    #print(len(data))
    nodeList = []
    for key, values in data.items():
        nodeList.extend(values)
    print(path,yuzhi,len(nodeList)/len(g.nodes()))
    communitys=readjson(filename_communitys)
    ret = defaultdict(list)
    if start!=0:
        simMatrix=np.load(path+'simMatrixHop' + str(start-1)+'_'+str(yuzhi) + '.npy')
        ret = readjson(path+'ret.json')
    else:
        simMatrix = np.array([[-1.0 for i in range(matrix_size)] for j in range(matrix_size)])
        #simMatrix=np.zeros([matrix_size,matrix_size])
    for hop in range(start, khop):
        if hop == 0:
            for label, nodeid in data.items():  # 选中的点（label：nodeid）
                ret[str(nodeid[0])] = list(set(range(matrix_size)).difference(set(communitys[label])))
                #ret[str(nodeid[0])] = random.sample(ret[str(nodeid[0])], int(len(ret[str(nodeid[0])]) ))
                for nodei in nodeid:
                    for nodej in ret[str(nodeid[0])]:
                        if Degrees[nodei]!=0 and Degrees[nodej]!=0:
                            simMatrix[nodei][nodej]=cacDistance(Degrees[nodei],Degrees[nodej])
                            print(simMatrix[nodei][nodej], nodei, nodej, path + str(hop))
        else:
            A = nx.adjacency_matrix(graph) ** hop + sp.csr_matrix(np.eye(len(graph)))
            for label, nodeid in data.items():  # 选中的点（label：nodeid）
                for nodei in nodeid:
                    for nodej in ret[str(nodeid[0])]:
                        if Degrees[nodei]!=0 and Degrees[nodej]!=0:
                            simMatrix[nodei][nodej] = simMatrix[nodei][nodej] + caculateSim(A, nodei, nodej, Degrees)
                            simMatrix[nodej][nodei] = simMatrix[nodei][nodej]
                            print(simMatrix[nodei][nodej], nodei, nodej,path+str(hop))
        np.save(path+'simMatrixHop' + str(hop)+'_'+str(yuzhi) + '.npy', simMatrix)
        savejson(ret,path+'ret.json')
    return graph
def generateSimMatrix_Pagerank(g,path,chooserate=0.5,start=0,khop=6):
    filename_node = path + 'node' + str(chooserate) + '.json'
    filename_communitys = path + 'communitys.json'
    matrix_size = len(g)
    data = readjson(path=filename_node)
    print(data)
    print(len(data))
    communitys=readjson(filename_communitys)
    simMatrix = np.array([[-1.0 for i in range(matrix_size)] for j in range(matrix_size)])
    G=g
    degrees = G.degree()
    ranks=pr(G)
    for label, nodeid in data.items():  # 选中的点（label：nodeid）
        ret=set(range(matrix_size)).difference(set(communitys[label]))
        for nodei in nodeid:
            for nodej in ret:
                if degrees[nodei]!=0 and degrees[nodej]!=0:
                    simMatrix[nodei][nodej] = cacDistance(ranks[nodei],ranks[nodej])
                    simMatrix[nodej][nodei] = simMatrix[nodei][nodej]
                    print(simMatrix[nodei][nodej], nodei, nodej)
                    if simMatrix[nodei][nodej]==0:
                        print(nodei,nodej)
    mkdir(path+'pagerank')
    np.save(path+'pagerank/simMatrixHop' +'_'+str(chooserate) + '.npy', simMatrix)
    return G
def mkdir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def generateSimMatrix_cos(g,path,chooserate=0.5):
    filename_node = path + 'node' + str(chooserate) + '.json'
    filename_communitys = path + 'communitys.json'
    graph = g
    Degrees = defaultdict(int)
    for i in range(len(graph)):
        degree = len(graph[i])
        Degrees[i] = degree
    matrix_size = len(graph)
    data = readjson(path=filename_node)
    G=graph
    G_ig = ig.Graph.from_networkx(G)

    # 计算节点特征
    degrees = G_ig.degree()
    max_degree = max(degrees)
    degree_centrality = [d / max_degree for d in degrees]
    closeness_centrality = G_ig.closeness()
    betweenness_centrality = G_ig.betweenness()
    eigenvector_centrality = G_ig.eigenvector_centrality()
    clustering_coeffs = G_ig.transitivity_local_undirected()
    ranks=list(pr(G).values())
    features=[]
    for node in G.nodes():
        features.append([
            degrees[node] + 1,
            clustering_coeffs[node],
            betweenness_centrality[node],
            degree_centrality[node],
            closeness_centrality[node],
            eigenvector_centrality[node],
            ranks[node]
        ])
    features = np.array(features)
    # 如果存在 NaN 值，将其替换为 0
    features = np.nan_to_num(features)

    # 对每个特征进行归一化
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    communitys=readjson(filename_communitys)
    simMatrix = np.array([[-1.0 for i in range(matrix_size)] for j in range(matrix_size)])
    for label, nodeid in data.items():  # 选中的点（label：nodeid）
        ret=set(range(matrix_size)).difference(set(communitys[label]))
        for nodei in nodeid:
            for nodej in ret:
                if Degrees[nodei]!=0 and Degrees[nodej]!=0:
                    simMatrix[nodei][nodej] =cosine(features[nodei],features[nodej])
                    simMatrix[nodej][nodei] =simMatrix[nodei][nodej]
                    print(simMatrix[nodei][nodej], nodei, nodej)
    mkdir(path + 'degree')
    np.save(path + 'degree/simMatrixHop' + '_' + str(chooserate) + '.npy', simMatrix)
    return graph
def caculateSim(A, i, j, Degrees):
    nodeseti = A.getrow(i).nonzero()[1]
    nodesetj = A.getrow(j).nonzero()[1]
    dseti = [Degrees[index] for index in nodeseti]
    dsetj = [Degrees[index] for index in nodesetj]
    #print(len(dseti),len(dsetj))
    sim = DWT(dseti, dsetj)
    # if cacDistance(len(dseti),len(dsetj))>5:
    #     sim=10000

    # else:
    #sim=DWT(dseti,dsetj)
    #print(sim[0],DWT(dseti, dsetj))
    return sim


def DWT(seti, setj):
    seti, setj = np.sort(seti), np.sort(setj)
    g = np.zeros([len(seti), len(setj)])
    for i in range(len(seti)):
        for j in range(len(setj)):
            g[i][j] = cacDistance(seti[i], setj[j])
    for i in range(len(seti)):
        for j in range(len(setj)):
            dij = cacDistance(seti[i], setj[j])
            if i == 0 and j != 0:
                g[i][j] = g[i][j - 1] + dij
            elif i != 0 and j == 0:
                g[i][j] = g[i - 1][j] + dij
            elif i == 0 and j == 0:
                g[i][j] = dij
            else:
                g[i][j] = min(g[i - 1][j] + dij, g[i - 1][j - 1] + 2 * dij, g[i][j - 1] + dij)
    return g[len(seti) - 1][len(setj) - 1]
def FASTDWT(seti,setj):
    seti, setj = np.sort(seti), np.sort(setj)
    seti=[[i] for i in seti]
    setj=[[j] for j in setj]
    distance,path=fastdtw(seti,setj,dist=cacDistance)
    return distance

def cacDistance(a, b):
    return max(a, b) / min(a, b) - 1
def cacWeight(path,chooserate=0.5):
    simMatrix = np.load(path+'simMatrixHop' + '_'+str(chooserate) + '.npy')
    for row in range(simMatrix.shape[0]):
        sumweight = 1
        for col in range(simMatrix.shape[1]):
            if simMatrix[row][col] !=-1:
                simMatrix[row][col] = exp(-simMatrix[row][col])
                sumweight = sumweight + simMatrix[row][col]
                #print(row,col)
            else:
                simMatrix[row][col]=0
        for col in range(simMatrix.shape[1]):
            if simMatrix[row][col] > 0:
                simMatrix[row][col] = simMatrix[row][col] / sumweight
                #print(row,col)
    np.save(path + 'weightMatrix' + str(chooserate) + '.npy', simMatrix)
    return simMatrix
def zscorenorm(matrix):

    matrix= (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    return matrix

def generateStructWeight(dir,args):
    weightList=[]
    for time in range(args.time_steps):
        subdir = dir + '{}/'.format(time)
        if os.path.exists(subdir + 'weightMatrix'+str(args.chooserate) + '.npy'):
            weight=np.load(subdir + 'weightMatrix'+str(args.chooserate) + '.npy')
        else:
            weight=cacWeight(subdir, chooserate=args.chooserate)

        weight = np.where(weight > calculateThreshold(weight, 100 - args.threshold), weight, 0)
        weight=normalize(weight)
        weightList.append(weight)
    return weightList


def generateWeight(dir,chooserate=1,loadcache=False,threshold=None,structure_type='pagerank'):
    Strucpath=dir+'{}/'.format(structure_type)
    Featurepath=dir+'feature/'
    if os.path.exists(Featurepath+ 'weightMatrix'+str(chooserate) + '.npy'):
        #FeatureWeight = cacWeight(Featurepath, chooserate=chooserate)
        FeatureWeight=np.load(Featurepath+ 'weightMatrix'+str(chooserate) + '.npy')
    else:
        FeatureWeight=cacWeight(Featurepath,chooserate=chooserate)
    if os.path.exists(Strucpath+ 'weightMatrix'+str(chooserate) + '.npy'):
        #StrucWeight = cacWeight(Strucpath, chooserate=chooserate)
        StrucWeight = np.load(Strucpath+ 'weightMatrix'+str(chooserate) + '.npy')
    else:
        StrucWeight = cacWeight(Strucpath, chooserate=chooserate)

    StrucWeight=np.where(StrucWeight>calculateThreshold(StrucWeight,100-threshold[0]),StrucWeight,0)
    FeatureWeight=np.where(FeatureWeight>calculateThreshold(FeatureWeight,100-threshold[1]),FeatureWeight,0)

    StrucWeight=normalize(StrucWeight)
    FeatureWeight=normalize(FeatureWeight)


    StrucWeight=torch.from_numpy(StrucWeight).float()
    FeatureWeight=torch.from_numpy(FeatureWeight).float()
    # if loadcache==True:
    #     Weight=np.load(path+'weightMatrix.npy')

    return StrucWeight, FeatureWeight

def calculateThreshold(weights,threshold):
    non_zero_weights = weights[weights != 0]
    return np.percentile(non_zero_weights, threshold)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
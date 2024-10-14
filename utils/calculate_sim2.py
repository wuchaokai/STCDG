import networkx as nx
from collections import defaultdict
import heapq
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import os
from math import exp
from utils.Pagerank import pr
from dataloader import evolve_edges
def get_second_order_neighbors(graph, node):
    first_order_neighbors = set(graph.neighbors(node))
    second_order_neighbors = set()

    for neighbor in first_order_neighbors:
        second_order_neighbors.update(graph.neighbors(neighbor))

    second_order_neighbors.update(first_order_neighbors)
    return set(second_order_neighbors)


def calculate_in_out_degree(graph, second_order_neighbors):

    in_deg = 0
    out_deg = 0
    for neighbor in second_order_neighbors:
        for n in graph.neighbors(neighbor):
            if n in second_order_neighbors:
                in_deg += 1
            else:
                out_deg += 1

    if in_deg==0:
        print(in_deg)
        print(out_deg)

    return out_deg/in_deg

def get_top_k_nodes(degrees_dict, k):
    #return [node for node, degree in heapq.nsmallest(k, degrees_dict.items(), key=lambda x: x[1])]
    return [node for node, degree in heapq.nsmallest(k, degrees_dict.items(), key=lambda x: x[1])]


def choose_oversquashed_nodes(graph,select_rate):
    if select_rate==1:
        return list(graph.nodes())
    score_list={}
    for node in graph:
        if graph.degree(node)==0:
            score_list[node]=1000000
            continue
        score_list[node]=calculate_in_out_degree(graph,get_second_order_neighbors(graph,node))

    node_list=get_top_k_nodes(score_list,max(int(select_rate*len(score_list)),1))

    return node_list


def find_anchor_points_spectral(G, select_rate=0.1):
    if select_rate==1:
        return list(G.nodes())
    adjacency_matrix = nx.to_numpy_array(G)
    n_clusters = max(1, int(select_rate*len(G.nodes)))
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    labels = spectral.fit_predict(adjacency_matrix)
    clusters = {}
    for node, label in zip(G.nodes, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node)
    anchor_points = []
    for cluster_nodes in clusters.values():
        representative_node = max(cluster_nodes, key=lambda node: G.degree(node))
        #representative_node=sorted(cluster_nodes, key=lambda node: G.degree(node), reverse=True)
        anchor_points.append(representative_node)
    return anchor_points


def shortest_path_distance_matrix(G,num_nodes):
    """
    计算图中任意两点之间的最短路径距离，并返回一个距离矩阵。

    参数:
    G (networkx.Graph): 输入的图。

    返回:
    distance_matrix (numpy.ndarray): 最短路径距离矩阵。
    """
    nodes = list(G.nodes())

    distance_matrix = np.zeros((num_nodes, num_nodes))

    for source in nodes:
        # 使用Dijkstra算法计算从source到所有其他节点的最短路径
        lengths = nx.single_source_dijkstra_path_length(G, source)
        for target in nodes:
            if source == target:
                distance_matrix[source][target] = 0
            else:
                distance_matrix[source][target] = lengths.get(target, np.inf)

    return distance_matrix


def degree_matrix(G,num_nodes):
    """
    计算图中每个节点的度数，并返回一个度数矩阵。

    参数:
    G (networkx.Graph): 输入的图。

    返回:
    degree_matrix (numpy.ndarray): 度数矩阵，每行一个节点的度数。
    """
    nodes = list(G.nodes())
    degree_matrix = np.zeros((num_nodes, 1), dtype=int)

    for node in nodes:
        degree_matrix[node] = G.degree(node)

    return degree_matrix


def complete_graph_with_all_nodes(G, num_nodes):
    """
    输入一个图和节点个数，确保输出一个包含所有指定节点的完整图。

    参数:
    G (networkx.Graph): 输入的图。
    num_nodes (int): 节点的总个数。

    返回:
    complete_G (networkx.Graph): 包含所有指定节点的完整图。
    """
    # 获取已有的所有节点
    existing_nodes = set(G.nodes())

    # 获取孤立的节点
    isolated_nodes = [node for node in range(num_nodes) if node not in existing_nodes]

    # 创建一个新的图
    complete_G = G.copy()

    # 添加孤立节点到图中
    for node in isolated_nodes:
        complete_G.add_node(node)

    return complete_G

def caculateSim(graph, i, j):
    first_order_neighborsi= set(graph.neighbors(i))
    first_order_neighborsj = set(graph.neighbors(j))
    dseti = [graph.degree(index) for index in first_order_neighborsi]
    dsetj = [graph.degree(index) for index in first_order_neighborsj]
    sim1=DWT(dseti, dsetj)

    second_order_neighborsi,second_order_neighborsj = set(),set()
    for neighbor in first_order_neighborsi:
        second_order_neighborsi.update(graph.neighbors(neighbor))
    for neighbor in first_order_neighborsj:
        second_order_neighborsj.update(graph.neighbors(neighbor))
    dseti = [graph.degree(index) for index in second_order_neighborsi]
    dsetj = [graph.degree(index) for index in second_order_neighborsj]
    sim2 = DWT(dseti, dsetj)
    return sim1+sim2

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

def cacDistance(a, b):
    return max(a, b) / min(a, b) - 1

def generateSimMatrix(graph,path,node_num,select_rate=0.5,match_rate=0.3,start=0,khop=2):

    os.makedirs(path, exist_ok=True)
    print('find over-squashed nodes')
    oversquashed_nodes=choose_oversquashed_nodes(graph,select_rate)
    print('find anchor nodes')
    matched_nodes=find_anchor_points_spectral(graph,match_rate)
    distance_matrix=shortest_path_distance_matrix(graph,node_num)

    simMatrix = np.array([[-1.0 for i in range(node_num)] for j in range(node_num)])


    for nodei in oversquashed_nodes:
        for nodej in matched_nodes:
            if distance_matrix[nodei][nodej]>2 and graph.degree(nodei)!=0 and graph.degree(nodej)!=0:
                simMatrix[nodei][nodej] = cacDistance(graph.degree(nodei), graph.degree(nodej)) + caculateSim(graph, nodei, nodej)
                simMatrix[nodej][nodei] = simMatrix[nodei][nodej]
                print(simMatrix[nodei][nodej], nodei, nodej, path)
    np.save(path + 'simMatrixHop_{}{}.npy'.format(str(select_rate),str(match_rate)), simMatrix)
    print('save simMatrixHop_{}{}.npy'.format(str(select_rate),str(match_rate)))
def generateSimMatrix_Pagerank(graph,path,node_num,select_rate=0.5,match_rate=0.3):
    os.makedirs(path, exist_ok=True)
    oversquashed_nodes = choose_oversquashed_nodes(graph, select_rate)
    matched_nodes = find_anchor_points_spectral(graph, match_rate)
    distance_matrix = shortest_path_distance_matrix(graph, node_num)

    simMatrix = np.array([[-1.0 for i in range(node_num)] for j in range(node_num)])

    ranks=pr(graph)
    for nodei in oversquashed_nodes:
        for nodej in matched_nodes:
            if distance_matrix[nodei][nodej] > 2 and graph.degree(nodei) != 0 and graph.degree(nodej) != 0:
                    simMatrix[nodei][nodej] = cacDistance(ranks[nodei],ranks[nodej])
                    simMatrix[nodej][nodei] = simMatrix[nodei][nodej]
                    print(simMatrix[nodei][nodej], nodei, nodej,path)

    np.save(path + 'simMatrixHop_{}{}.npy'.format(str(select_rate), str(match_rate)), simMatrix)
    print('save simMatrixHop_{}{}.npy'.format(str(select_rate), str(match_rate)))


def retain_values(matrix, row_indices, col_indices):
    """
    保持行索引和列索引同时覆盖到的数值不变，其他值清0。

    参数:
    matrix (np.ndarray): 原始矩阵
    row_indices (list of int): 行索引列表
    col_indices (list of int): 列索引列表

    返回:
    np.ndarray: 新矩阵，其中只有指定位置的值保留，其余值为零
    """
    # 创建一个与原始矩阵大小相同的全零矩阵
    new_matrix = np.ones_like(matrix)*(-1)

    # 保持行索引和列索引同时覆盖到的数值不变
    for i in row_indices:
        for j in col_indices:
            new_matrix[i, j] = matrix[i, j]

    return new_matrix
def cacWeight(graph,path,select_rate=0.5,match_rate=0.1):
    simMatrix = np.load(path+'simMatrixHop_11.npy')
    oversquashedNodes=sorted(choose_oversquashed_nodes(graph,select_rate))
    matchNodes=sorted(find_anchor_points_spectral(graph,match_rate))
    simMatrix=retain_values(simMatrix,oversquashedNodes,matchNodes)
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
    np.save(path + 'weightMatrix_{}{}.npy'.format(str(select_rate),str(match_rate)) , simMatrix)
    return simMatrix

def generateStructWeight(dir,args,graphs):
    weightList=[]
    for time in range(args.time_steps):
        subdir = dir + '{}/'.format(time)
        if os.path.exists(subdir + 'weightMatrix_{}{}.npy'.format(str(args.select_rate),str(args.match_rate))):
            weight=np.load(subdir + 'weightMatrix_{}{}.npy'.format(str(args.select_rate),str(args.match_rate)))
        else:
            weight=cacWeight(graphs[time],subdir, select_rate=args.select_rate,match_rate=args.match_rate)
        #weight = cacWeight(graphs[time], subdir, select_rate=args.select_rate, match_rate=args.match_rate)
        weight = np.where(weight > calculateThreshold(weight, 100 - args.threshold), weight, 0)
        weight=normalize(weight)
        weightList.append(weight)
    return weightList

def calculateThreshold(weights,threshold):
    non_zero_weights = weights[weights != 0]
    return np.percentile(non_zero_weights, threshold)

def normalize(mx):
    rowsum = mx.sum(axis=1)
    rowsum[rowsum == 0] = 1  # 处理行和为零的情况，将其设为1，避免除以零
    mx_normalized = mx / rowsum[:, np.newaxis]  # 对每行除以对应的和，并确保是二维数组
    return mx_normalized



def visualize_graph_with_anchor_points(G, anchor_points):
    """
    可视化图和锚点。

    参数:
    G (networkx.Graph): 输入的图。
    anchor_points (list): 锚点列表。
    """
    pos = nx.spring_layout(G)
    labels = {node: G.nodes[node].get('label', node) for node in G.nodes}
    colors = ['black' if node in anchor_points else 'red' for node in G.nodes]

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, node_color=colors, with_labels=True, node_size=300, cmap=plt.cm.rainbow)
    nx.draw_networkx_nodes(G, pos, nodelist=anchor_points, node_color='blue', node_size=500)
    plt.show()


def compute_neighbor_degrees(adj):
    """
    计算每个节点的度数列表（包括自身、一阶和二阶邻居）。

    参数：
    adj -- 稀疏邻接矩阵（scipy.sparse 矩阵）

    返回：
    zero_order_degrees -- 自身度数列表
    first_order_degrees -- 一阶邻居度数列表
    second_order_degrees -- 二阶邻居度数列表
    """
    # 转换为 NetworkX 图
    G = nx.from_scipy_sparse_matrix(adj)

    # 初始化存储自身、一阶和二阶邻居度数的列表
    zero_order_degrees = []
    first_order_degrees = []
    second_order_degrees = []

    # 计算自身、一阶和二阶邻居度数
    for node in G.nodes():
        # 自身度数
        zero_degree = G.degree(node)
        zero_order_degrees.append(zero_degree)

        # 一阶邻居度数列表
        first_neighbors = list(G.neighbors(node))
        first_degrees = [G.degree(neighbor) for neighbor in first_neighbors]
        first_order_degrees.append(first_degrees)

        # 二阶邻居度数列表
        second_neighbors = set()
        for neighbor in first_neighbors:
            second_neighbors.update(G.neighbors(neighbor))
        #second_neighbors.discard(node)  # 排除节点自身
        second_degrees = [G.degree(neighbor) for neighbor in second_neighbors]
        second_order_degrees.append(second_degrees)

    return zero_order_degrees, first_order_degrees, second_order_degrees

def caculateTimeSim(adj_list,num_nodes,path):
    evolve_matrix_list=evolve_edges(adj_list,num_nodes)
    Z,F,S=[],[],[]
    for time in range(len(adj_list)):
        zero_order_degrees, first_order_degrees, second_order_degrees=compute_neighbor_degrees(evolve_matrix_list[time])
        Z.append(zero_order_degrees)
        F.append(first_order_degrees)
        S.append(second_order_degrees)
    simMatrix=np.ones((len(adj_list),num_nodes,len(adj_list)))*(-1)
    for i in range(1,len(adj_list)):
        for j in range(i):
            zero_order_degreesi, first_order_degreesi, second_order_degreesi=Z[i],F[i],S[i]
            zero_order_degreesj, first_order_degreesj, second_order_degreesj = Z[j], F[j], S[j]
            for node in range(num_nodes):
                if zero_order_degreesj[node]!=0 and zero_order_degreesi[node]!=0:
                    simMatrix[i][node][j]=cacDistance(zero_order_degreesi[node],zero_order_degreesj[node])
                    simMatrix[i][node][j]=simMatrix[i][node][j]+DWT(first_order_degreesi[node],first_order_degreesj[node])+DWT(second_order_degreesi[node],second_order_degreesj[node])
                    print(node,i,j,simMatrix[i][node][j])
    np.save(path + 'simMatrixHop_time.npy', simMatrix)


def cacTimeWeight(path):
    simMatrix = np.load(path+'simMatrixHop_time.npy')


    for time in range(simMatrix.shape[0]):
        for node in range(simMatrix.shape[1]):
            sumweight=1
            for step in range(simMatrix.shape[2]):
                if simMatrix[time][node][step]!=-1:
                    simMatrix[time][node][step]=exp(-simMatrix[time][node][step])
                    sumweight = sumweight + simMatrix[time][node][step]
                else:
                    simMatrix[time][node][step] =0
            for step in range(simMatrix.shape[2]):
                if simMatrix[time][node][step]>0:
                    simMatrix[time][node][step]=simMatrix[time][node][step]/sumweight
    np.save(path + 'weightMatrix_time.npy', simMatrix)
    return simMatrix

def generateTimeWeight(path,args):
    weightList=[]
    for time in range(args.time_steps):
        if os.path.exists(path + 'weightMatrix_time.npy'.format(str(args.select_rate),str(args.match_rate))):
            weight=np.load(path + 'weightMatrix_time.npy'.format(str(args.select_rate),str(args.match_rate)))
        else:
            weight=cacTimeWeight(path)

        weight = np.where(weight > calculateThreshold(weight, 100 - args.threshold), weight, 0)
        #weight=normalize(weight)
    return weight
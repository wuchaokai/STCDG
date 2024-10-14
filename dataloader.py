import numpy as np
import networkx as nx
import pickle as pkl
import torch
from sklearn.model_selection import train_test_split
from utils.sample import run_random_walks_n2v
import os
from scipy.sparse import csr_matrix
def load_graphs(dataset):
    # Load graph snapshots given the name of dataset
    with open("./data/{}/{}".format(dataset, "graph.pkl"), "rb") as f:
        graphs = pkl.load(f)
    print("Loaded {} graphs ".format(len(graphs)))
    adjs = [nx.adjacency_matrix(g) for g in graphs]
    feats=[np.eye(len(graphs[-1].nodes())) for _ in range(len(adjs))]
    return graphs, adjs,feats



def load_graphs2(dataset):
    data=torch.load('data/{}/{}.data'.format(dataset,dataset))
    edge_index_list=data['edge_index_list']
    if dataset=='as733':
        edge_index_list=edge_index_list
    # min_value = min(edge_index.min().item() for edge_index in edge_index_list)
    # edge_index_list=[edge_index - min_value for edge_index in edge_index_list]
    graphs=[edge_index_to_networkx(edge_index_list[t]) for t in range(len(edge_index_list))]
    adjs=[nx.adjacency_matrix(g) for g in graphs]
    node_num=data['num_nodes']
    feats=[np.eye(node_num) for _ in range(len(adjs))]
    return graphs,adjs,feats

def edge_index_to_networkx(edge_index, num_nodes=None):
    """
    将PyTorch Geometric的edge_index转换为NetworkX图。

    参数:
    - edge_index: torch.Tensor, 形状为 [2, num_edges] 的边索引张量
    - num_nodes: int, 节点总数（可选，如果不提供，将根据edge_index计算）

    返回:
    - G: networkx.Graph, 转换后的NetworkX图
    """
    G = nx.Graph()

    # 添加边
    edges = edge_index.t().tolist()  # 转置并转换为列表
    G.add_edges_from(edges)

    # 添加节点，如果提供了num_nodes，则确保所有节点都在图中
    if num_nodes is not None:
        G.add_nodes_from(range(num_nodes))

    return G



def get_evaluation_data(graphs,args):
    """ Load train/val/test examples to evaluate link prediction performance"""
    eval_idx = len(graphs) - 2
    eval_graph = graphs[eval_idx]
    next_graph = graphs[eval_idx+1]
    #print("Generating eval data ....")
    path='data/'+args.dataset+'/'+args.evaluation_data_filename
    if args.sample_type=='random':
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            create_data_splits(eval_graph, next_graph, val_mask_fraction=0.2,
                               test_mask_fraction=0.6)
        np.savez(path, train_edges=train_edges, train_edges_false=train_edges_false, val_edges=val_edges,
                 val_edges_false=val_edges_false, test_edges=test_edges, test_edges_false=test_edges_false)
    elif os.path.exists(path):
        #print("Load evaluation data")
        data=np.load(path)
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false=data['train_edges'],data['train_edges_false'],data['val_edges'],data['val_edges_false'],data['test_edges'],data['test_edges_false']
    else:
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        create_data_splits(eval_graph, next_graph, val_mask_fraction=0.2,
                            test_mask_fraction=0.6)
        np.savez(path, train_edges=train_edges, train_edges_false=train_edges_false, val_edges=val_edges,val_edges_false=val_edges_false,test_edges=test_edges,test_edges_false=test_edges_false)
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def create_data_splits(graph, next_graph, val_mask_fraction=0.2, test_mask_fraction=0.6):
    edges_next = np.array(list(nx.Graph(next_graph).edges()))
    edges_positive = []  # Constraint to restrict new links to existing nodes.
    for e in edges_next:
        if graph.has_node(e[0]) and graph.has_node(e[1]):
            edges_positive.append(e)
    edges_positive = np.array(edges_positive)  # [E, 2]
    edges_negative = negative_sample(edges_positive, graph.number_of_nodes(), next_graph)

    train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(edges_positive,
                                                                            edges_negative,
                                                                            test_size=val_mask_fraction + test_mask_fraction)
    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos,
                                                                                    test_neg,
                                                                                    test_size=test_mask_fraction / (
                                                                                                test_mask_fraction + val_mask_fraction))

    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg


def negative_sample(edges_pos, nodes_num, next_graph):
    edges_neg = []
    while len(edges_neg) < len(edges_pos):
        idx_i = np.random.randint(0, nodes_num)
        idx_j = np.random.randint(0, nodes_num)
        if idx_i == idx_j:
            continue
        if next_graph.has_edge(idx_i, idx_j) or next_graph.has_edge(idx_j, idx_i):
            continue
        if edges_neg:
            if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg:
                continue
        edges_neg.append([idx_i, idx_j])
    return edges_neg

def get_context_pairs(graphs, adjs):
    """ Load/generate context pairs for each snapshot through random walk sampling."""
    print("Computing training pairs ...")
    context_pairs_train = []
    for i in range(len(graphs)):
        context_pairs_train.append(run_random_walks_n2v(graphs[i], adjs[i], num_walks=10, walk_len=20))

    return context_pairs_train

def csr_to_edge_index(csr_matrix):
    """
    将CSR格式的稀疏矩阵转换为PyTorch中的edge_index格式。

    参数:
    csr_matrix (scipy.sparse.csr_matrix): CSR格式的稀疏矩阵。

    返回:
    torch.Tensor: PyTorch的edge_index格式。
    """
    # 将CSR矩阵转换为COO（坐标格式）矩阵
    coo_matrix = csr_matrix.tocoo()

    # 提取行索引和列索引
    row = torch.tensor(coo_matrix.row, dtype=torch.long)
    col = torch.tensor(coo_matrix.col, dtype=torch.long)

    # 构建 edge_index
    edge_index = torch.stack([row, col], dim=0)

    return edge_index

def load_feature(args):
    if args.trainable_feat:
        x = None
    else:
        if args.pre_defined_feature is not None:
            import scipy.sparse as sp
            # if args.dataset == 'disease':
            #     feature = sp.load_npz(disease_path).toarray()
            x = torch.from_numpy(feature).float().to(args.device)
            logger.info('using pre-defined feature')
        else:
            self.x = torch.eye(args.num_nodes).to(args.device)
            logger.info('using one-hot feature')
        args.nfeat = self.x.size(1)


def accumulate_edges(adj_list,num_nodes):

    cumulative_matrix = csr_matrix((num_nodes, num_nodes))

    # 用于存储每个时刻的累积邻接矩阵的列表
    new_adj_list = []

    # 遍历每个时刻的邻接矩阵
    for adj_matrix in adj_list:
        adj_matrix.data=np.clip(adj_matrix.data, 0, 1)
        # 累加当前时刻的邻接矩阵
        cumulative_matrix += adj_matrix

        # 将累积矩阵添加到列表中
        cumulative_matrix.data = np.clip(cumulative_matrix.data, 0, 1)
        new_adj_list.append(cumulative_matrix.copy())

    return new_adj_list

def evolve_edges(adj_list,num_nodes):
    evolve_matrix = csr_matrix((num_nodes, num_nodes))
    evolve_matrix_list=[]
    for adj_matrix in adj_list:
        adj_matrix.data=np.clip(adj_matrix.data, 0, 1)
        # 累加当前时刻的邻接矩阵
        evolve_matrix=extract_unique_edges(evolve_matrix,adj_matrix)
        # 将累积矩阵添加到列表中
        evolve_matrix_list.append(evolve_matrix.copy())
    return evolve_matrix_list


def extract_unique_edges(a1, a2):
    """
    提取第二个稀疏矩阵中不在第一个稀疏矩阵中的边，并构成新的稀疏矩阵。

    参数:
    a1 (csr_matrix): 第一个稀疏矩阵
    a2 (csr_matrix): 第二个稀疏矩阵

    返回:
    csr_matrix: 只包含在a2中但不在a1中的边的新稀疏矩阵
    """
    # 获取稀疏矩阵的形状
    shape = a1.shape

    # 将稀疏矩阵转换为边列表
    a1_edges = set(zip(a1.nonzero()[0], a1.nonzero()[1]))
    a2_edges = set(zip(a2.nonzero()[0], a2.nonzero()[1]))

    # 提取出 a2 中不在 a1 中的边
    changed_edges = a2_edges - a1_edges

    # 构建新的稀疏矩阵
    if len(changed_edges) == 0:
        # 如果没有变化的边，返回一个空的稀疏矩阵
        return csr_matrix((shape))

    row_indices = np.array([edge[0] for edge in changed_edges])
    col_indices = np.array([edge[1] for edge in changed_edges])
    data = np.ones(len(changed_edges))

    new_sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=shape)
    return new_sparse_matrix


def include_isolated_nodes(original_matrix, num_nodes):
    """
    调整稀疏矩阵的形状，以包含所有节点，包括孤立节点。

    参数:
    - original_matrix: csr_matrix, 原始稀疏矩阵，只包含有边的节点
    - num_nodes: int, 包括孤立节点在内的总节点数

    返回:
    - new_matrix: csr_matrix, 调整后的稀疏矩阵，包含所有节点
    """
    # 获取原始矩阵的数据、行索引和列索引
    data = original_matrix.data
    rows = original_matrix.indices
    row_indptr = original_matrix.indptr

    # 创建新的行指针数组，长度为 num_nodes + 1
    new_row_indptr = np.zeros(num_nodes + 1, dtype=int)

    # 填充新的行指针数组
    new_row_indptr[:len(row_indptr)] = row_indptr
    for i in range(len(row_indptr) - 1, len(new_row_indptr) - 1):
        new_row_indptr[i + 1] = new_row_indptr[i]

    # 创建新的稀疏矩阵，调整形状
    new_matrix = csr_matrix((data, rows, new_row_indptr), shape=(num_nodes, num_nodes))

    return new_matrix


def csr_to_adj_tensor(csr_matrix, num_nodes):
    """
    将 SciPy 的 CSR 矩阵转换为 PyTorch 格式的邻接矩阵。

    参数:
    csr_matrix (scipy.sparse.csr_matrix): 输入的 CSR 矩阵。
    num_nodes (int): 节点的个数。

    返回:
    torch.sparse.FloatTensor: 转换后的 PyTorch 稀疏张量。
    """
    coo = csr_matrix.tocoo()  # 将 CSR 矩阵转换为 COO 格式
    values = coo.data
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
    shape = (num_nodes, num_nodes)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    return indices
import matplotlib.pyplot as plt

# # 示例数据
# x = [2, 3, 4, 5,6]
# y1 = [0.5710, 0.5896, 0.6018, 0.5945, 0.5871]  # 第一组数据
# y2 = [0.7050, 0.7371, 0.7645, 0.7720, 0.8016]   # 第二组数据
#
# # 绘制两条曲线
# plt.plot(x, y1, label='EvolveGCN with Compensation', color='b', marker='o')
# plt.plot(x, y2, label='EvolveGCN', color='r', marker='x')
#
# # 添加图例
# plt.legend()
#
# # 添加标题和坐标轴标签
# plt.title('两组数据的曲线图')
# plt.xlabel('time step')
# plt.ylabel('AUC')
#
# # 显示图表
# plt.show()
from itertools import permutations
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import networkx as nx
from dataloader import load_graphs2,load_graphs
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import random
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
def get_time_fig():
    loaded_embeddings = torch.load('data/dblp/NoCompensate/embed_10.pt')[:-1]
    # 转换为 NumPy 数组
    id = 61
    A = np.array([emb[id].cpu().tolist() for emb in loaded_embeddings])

    # 计算余弦相似度
    similarity_matrix = cosine_similarity(A)
    # similarity_matrix = similarity_matrix ** 2
    # 标准化相似度矩阵
    similarity_matrix = (similarity_matrix - np.min(similarity_matrix)) / (
                np.max(similarity_matrix) - np.min(similarity_matrix))

    last_embedding_similarity = similarity_matrix[-1][::-1].reshape(1, -1)  # 包括自己

    # 绘制热图
    plt.figure(figsize=(20, 2))
    sns.heatmap(
        last_embedding_similarity,
        cmap='YlGn',  # 选择一个合适的颜色映射
        annot=True,  # 显示数值
        fmt='.4f',  # 数值格式
        annot_kws={"size": 32, 'weight': 'bold'},
        xticklabels=[],
        yticklabels=[]
    )
    # plt.xticks(ticks=np.arange(len(A))+ 0.5, labels=[f' {9-i}' for i in range(len(A))])
    # plt.yticks(ticks=np.arange(1)+ 0.5, labels=[9], rotation=0)

    plt.show()


def cosine_similarity_edge(a, b):
    """
    计算两个相同形状的向量组之间的余弦相似度。

    参数:
    a: numpy.ndarray，形状为 (n, m) 的向量组
    b: numpy.ndarray，形状为 (n, m) 的向量组

    返回:
    numpy.ndarray，形状为 (n,) 的余弦相似度数组
    """
    # 确保输入为 numpy 数组
    a = np.array(a)
    b = np.array(b)

    # 计算点积
    dot_product = np.sum(a * b, axis=1)

    # 计算范数
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    # 计算余弦相似度
    similarity = dot_product / (norm_a * norm_b)

    return similarity


def getST_fig():
    loaded_embeddings = torch.load('data/dblp/spatialTimeCompensate/embed.pt')[:-1]
    # 转换为 NumPy 数组
    id1=47
    id2=127
    A = np.array([emb[id1].cpu().tolist() for emb in loaded_embeddings])
    B=np.array([emb[id2].cpu().tolist() for emb in loaded_embeddings])
    similarity=cosine_similarity_edge(A,B)
    print(similarity)
    loaded_embeddings = torch.load('data/dblp/timeCompensate/embed_10.pt')[:-1]
    A = np.array([emb[id1].cpu().tolist() for emb in loaded_embeddings])
    B = np.array([emb[id2].cpu().tolist() for emb in loaded_embeddings])
    similarity=cosine_similarity_edge(A, B)
    print(similarity)
def get_spatial_fig_noCompensate():

    loaded_embeddings = torch.load('data/dblp/NoCompensate/embed_10.pt')[-2]
    graphs, _, _ = load_graphs2('dblp')
    # 转换为 NumPy 数组
    id = 196
    neighborId=[]
    for dis in range(10):
        neighborId.append(get_random_node_at_distance(graphs[-2],id,dis))


    A=np.array([loaded_embeddings[nid].cpu().tolist() for nid in neighborId])
    # 计算余弦相似度
    similarity_matrix = cosine_similarity(A)
    # similarity_matrix = similarity_matrix ** 2
    # 标准化相似度矩阵
    similarity_matrix = (similarity_matrix - np.min(similarity_matrix)) / (
            np.max(similarity_matrix) - np.min(similarity_matrix))

    #last_embedding_similarity = similarity_matrix[-1][::-1].reshape(1, -1)  # 包括自己

    # 绘制热图
    plt.figure(figsize=(8, 4))
    sns.heatmap(
        similarity_matrix,
        cmap='YlGn',  # 选择一个合适的颜色映射
        annot=True,  # 显示数值
        fmt='.2f',  # 数值格式
        annot_kws={"size": 10, 'weight': 'bold'},
        xticklabels=[],
        yticklabels=[]
    )
    # plt.xticks(ticks=np.arange(len(A))+ 0.5, labels=[f' {9-i}' for i in range(len(A))])
    # plt.yticks(ticks=np.arange(1)+ 0.5, labels=[9], rotation=0)

    plt.show()


def get_spatial_fig_Compensate():
    loaded_embeddings = torch.load('data/dblp/spatialCompensate/embed.pt')[-2]
    graphs, _, _ = load_graphs2('dblp')

    # 转换为 NumPy 数组
    id = 104
    neighborId=[]
    # for dis in range(10):
    #     neighborId.append(get_random_node_at_distance(graphs[-2],id,dis))
    # paths=find_paths(graphs[-2],id,10)
    # for i in range(len(paths)):
    #     if len(paths[i])>9:
    #         neighborId=paths[i+16]
    #         break
    # print(neighborId)
    neighborId.extend([217,152,154,206,49,75,93,311,305,104, 157, 192, 73, 308, 234, 252, 95, 91, 18])

    A=np.array([loaded_embeddings[nid].cpu().tolist() for nid in neighborId])
    # 计算余弦相似度
    similarity_matrix = cosine_similarity(A)
    # similarity_matrix = similarity_matrix ** 2
    # 标准化相似度矩阵
    similarity_matrix = (similarity_matrix - np.min(similarity_matrix)) / (
            np.max(similarity_matrix) - np.min(similarity_matrix))

    target_similarity = similarity_matrix[9].reshape(1, -1)  # 包括自己
    colors = plt.cm.Blues(target_similarity)[0]
    #colors = plt.cm.Blues(np.linspace(0, 1, 10))
    custom_cmap = ListedColormap(colors)
    # 绘制热图
    plt.figure(figsize=(40, 2))
    sns.heatmap(
        target_similarity,
        cmap='Blues',  # 选择一个合适的颜色映射
        annot=True,  # 显示数值
        fmt='.2f',  # 数值格式
        annot_kws={"size": 30, 'weight': 'bold'},
        xticklabels=[],
        yticklabels=[]
    )
    # plt.xticks(ticks=np.arange(len(A))+ 0.5, labels=[f' {9-i}' for i in range(len(A))])
    # plt.yticks(ticks=np.arange(1)+ 0.5, labels=[9], rotation=0)

    plt.show()

def get_spatial_fig_Compensate_graph():

    loaded_embeddings = torch.load('data/dblp/NoCompensate/embed.pt')[-2]
    graphs, _, _ = load_graphs2('dblp')
    # 转换为 NumPy 数组
    id = 104
    pathlen=10
    pathnum=2
    neighborId=[]
    neighborId_list=[]
    neighborId_list.append(id)
    embedding_groups=[]
    embedding_groups.append(loaded_embeddings[id].cpu().tolist())
    #for path in range(pathnum):
        # for dis in range(1,pathlen):
        #     neighborId.append(get_node_at_distance(graphs[-2],id,dis,path))
        #     neighborId_list.append(get_node_at_distance(graphs[-2],id,dis,path))
    neighborId_list=[104, 300, 192, 273, 25, 241, 87, 45, 152, 86, 305, 307, 72, 227, 118, 252, 89, 21, 207]


    print(neighborId)
    embedding_groups=[loaded_embeddings[nid].cpu().tolist() for nid in neighborId_list]
    # embedding_group_1 = np.array([[1, 0, 0],
    #                               [0, 1, 0],
    #                               [0, 0, 1]])
    #
    # embedding_group_2 = np.array([[1, 0, 0],
    #                               [0, 1, 1],
    #                               [1, 0, 1]])
    # embedding_groups = [embedding_group_1, embedding_group_2]
    create_graph_with_multiple_embeddings(embedding_groups,neighborId_list,pathnum, 0)
    exit()
    # 计算余弦相似度
    similarity_matrix = cosine_similarity(A)
    # similarity_matrix = similarity_matrix ** 2
    # 标准化相似度矩阵
    # similarity_matrix = (similarity_matrix - np.min(similarity_matrix)) / (
    #         np.max(similarity_matrix) - np.min(similarity_matrix))

    similarity_matrix = similarity_matrix[0].reshape(1, -1)  # 包括自己

    # 绘制热图
    plt.figure(figsize=(8, 4))
    sns.heatmap(
        similarity_matrix,
        cmap='YlGn',  # 选择一个合适的颜色映射
        annot=True,  # 显示数值
        fmt='.2f',  # 数值格式
        annot_kws={"size": 10, 'weight': 'bold'},
        xticklabels=[],
        yticklabels=[]
    )
    # plt.xticks(ticks=np.arange(len(A))+ 0.5, labels=[f' {9-i}' for i in range(len(A))])
    # plt.yticks(ticks=np.arange(1)+ 0.5, labels=[9], rotation=0)

    plt.show()
def get_nodes_at_distance(graph, node_id, distance):
    """
    获取指定节点到其他节点的特定距离的节点列表。

    :param graph: networkx.Graph 对象
    :param node_id: 节点 ID
    :param distance: 距离
    :return: 距离为指定值的节点列表
    """
    # 计算从指定节点到其他节点的最短路径长度
    shortest_paths = nx.single_source_shortest_path_length(graph, node_id)

    # 返回距离为指定值的节点列表
    return [node for node, dist in shortest_paths.items() if dist == distance]
def get_random_node_at_distance(graph, node_id, distance):
    """
    从距离指定节点的节点列表中随机抽取一个节点。

    :param graph: networkx.Graph 对象
    :param node_id: 节点 ID
    :param distance: 距离
    :return: 随机抽取的节点或 None（如果没有找到符合条件的节点）
    """
    #random.seed(2024)
    #np.random.seed(2023)
    nodes_at_distance = get_nodes_at_distance(graph, node_id, distance)
    if nodes_at_distance:
        return random.choice(nodes_at_distance)
    else:
        return None

def get_node_at_distance(graph, node_id, distance,index):
    """
    从距离指定节点的节点列表中随机抽取一个节点。

    :param graph: networkx.Graph 对象
    :param node_id: 节点 ID
    :param distance: 距离
    :return: 随机抽取的节点或 None（如果没有找到符合条件的节点）
    """
    #random.seed(2024)
    #np.random.seed(2023)
    nodes_at_distance = get_nodes_at_distance(graph, node_id, distance)
    if nodes_at_distance:
        if distance!=0:
            return nodes_at_distance[index]
        else:
            return nodes_at_distance[0]
    else:
        return None


def find_paths(graph, start_node, max_distance):
    paths = []
    current_level = {start_node: [start_node]}  # 初始化当前层级节点

    for distance in range(1, max_distance + 1):
        next_level = {}
        for node, path in current_level.items():
            neighbors = list(graph.neighbors(node))
            for neighbor in neighbors:
                if neighbor not in path:  # 确保不形成回路
                    new_path = path + [neighbor]
                    # 检查新节点的最短距离
                    if nx.shortest_path_length(graph, start_node, neighbor) == distance:
                        next_level[neighbor] = new_path

        current_level = next_level
        # 记录当前距离的路径
        paths.extend(current_level.values())

    return paths
def invert_dict(original_dict):
    inverted_dict = {}
    for key, value in original_dict.items():
        if value not in inverted_dict:
            inverted_dict[value] = [key]  # 如果值没有作为键，初始化为列表
        else:
            inverted_dict[value].append(key)  # 如果值已存在，添加到列表中
    return inverted_dict
def create_graph_with_multiple_embeddings(embedding_groups, id_list,pathnum,target_node_id):
    """
    创建图并根据节点相似度为节点上色，并按顺序连接多个嵌入组的节点到同一个目标节点。

    :param embedding_groups: 包含多个节点嵌入的列表（每个嵌入组都是一个 numpy 数组）
    :param target_node_id: 目标节点 ID
    """
    G = nx.Graph()
    for id in id_list:
        G.add_node(id)
    start_id=id_list[0]
    id_list=id_list[1:]
    for i in range(len(id_list)):
        if i % (len(id_list)/pathnum) !=0:
            G.add_edge(id_list[i-1],id_list[i])
        else:
            G.add_edge(start_id,id_list[i])

    # 计算相似度矩阵并获取目标节点的相似度
    similarity_matrix = cosine_similarity(embedding_groups)
    target_similarity = similarity_matrix[target_node_id]

    # 根据相似度设置节点颜色
    colors = plt.cm.Blues(target_similarity)  # 使用 viridis 颜色映射

    # 创建新的图形和坐标轴
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制图
    pos = nx.spring_layout(G)  # 节点布局
    nx.draw(G, pos, node_color=colors, with_labels=True, font_weight='bold', node_size=500, ax=ax, edge_color='gray')

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='Blues',
                               norm=plt.Normalize(vmin=min(target_similarity), vmax=max(target_similarity)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Cosine Similarity')

    ax.set_title(f'Node Similarity to Node {start_id}')
    plt.show()


def plot_gradient_bar_and_get_color():
    """
    绘制从 lower_bound 到 upper_bound 的渐变条形图，并返回输入值对应的颜色。

    参数:
    lower_bound: float，范围下限
    upper_bound: float，范围上限
    input_value: float，输入数值，范围在 [lower_bound, upper_bound]

    返回:
    str，输入值对应的颜色的十六进制表示
    """
    lower_bound=0
    upper_bound=100
    input_value=0.4108
    # 确保输入值在范围内
    if input_value < lower_bound or input_value > upper_bound:
        raise ValueError(f"Input value must be between {lower_bound} and {upper_bound}.")

        # 生成数值
    values = np.linspace(lower_bound, upper_bound, num=10)

    # 创建颜色映射
    norm = mcolors.Normalize(vmin=lower_bound, vmax=upper_bound)
    cmap = plt.get_cmap("YlGn")
    colors = cmap(norm(values))

    # 绘制条形图
    plt.bar(range(len(values)), values, color=colors)

    # 添加标签
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Gradient Color Bar from {}% to {}%'.format(lower_bound * 100, upper_bound * 100))

    # 设置 y 轴范围
    plt.ylim(0, upper_bound + 0.1)

    # 显示颜色条
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Value')

    # 获取输入值对应的颜色
    color = cmap(norm(input_value))
    hex_color = mcolors.to_hex(color)

    # 在图上显示对应颜色的矩形
    plt.gca().add_patch(plt.Rectangle((0.8, upper_bound * 0.5), 0.5, 0.5, color=color, ec='black'))
    plt.text(0.85, upper_bound * 0.5 + 0.05, f'Color: {hex_color}', fontsize=10, va='center')

    plt.show()

    return hex_color
def plot_embedding():
    dataset='as733'
    model='WinGNN'
    #loaded_embeddings = torch.load('data/{}/spatialTimeCompensate/embed_RTGCN1728311983.pt'.format(dataset,model))
    loaded_embeddings = torch.load('data/{}/NoCompensate/embed_WinGNN1728615682.pt'.format(dataset, model))
    if dataset in ['uci']:
        graphs, adjs, feats = load_graphs(dataset)
    elif dataset in ['as733', 'dblp', 'enron10']:
        graphs, adjs, feats = load_graphs2(dataset)
    i,u=[],[]
    path = 'data/' + dataset + '/evaluation_data.npz'
    data = np.load(path)
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false=data['train_edges'],data['train_edges_false'],data['val_edges'],data['val_edges_false'],data['test_edges'],data['test_edges_false']
    edges_pos,edges_false=[],[]
    edges_false.extend(train_edges_false)
    edges_false.extend(val_edges_false)
    edges_false.extend(test_edges_false)
    edges_pos.extend(train_edges)
    edges_pos.extend(val_edges)
    edges_pos.extend(test_edges)
    for edge in graphs[-1].edges():
        i.append(loaded_embeddings[edge[0]])
        u.append(loaded_embeddings[edge[1]])
    for edge in edges_false:
        i.append(loaded_embeddings[edge[0]])
        u.append(loaded_embeddings[edge[1]])
    emb=np.multiply(i,u)
    emb=StandardScaler().fit_transform(emb)
    labels=[1]*len(graphs[-1].edges())+[0]*len(edges_false)
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42,perplexity=50)
    features_tsne = tsne.fit_transform(emb)

    # 绘制t-SNE图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='jet', alpha=0.7)

    # 添加颜色条
    plt.colorbar(scatter)

    # 添加标题和标签
    plt.title('t-SNE Visualization of Features dataset:{} model:{}'.format(dataset,model))
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # 显示图形
    plt.show()

def plotParameter():

    x_values = [0.1,0.3,0.5,0.7,0.8,1]
    #x_values=[0.1,0.3,0.5,0.7,1]
    dataset='DBLP'
    y_values = [0.6344, 0.6478, 0.6127, 0.6390, 0.6826,0.7941]
    y_values2=[0.6168,0.6214,0.6150,0.6647,0.6954,0.8158]
    y_values3 = [0.5768, 0.6114, 0.5528, 0.6518, 0.6753,0.7070]
    y_values4=[0.6427,0.6637,0.6493,0.6763,0.6940,0.7891]
    y_values5 = [0.7647, 0.7665, 0.7806, 0.8114, 0.8221,0.8887]
    y_values6=[0.5955, 0.6703,	0.6778,	0.6746,	0.7037,	0.8149]
    #plt.figure(figsize=(20,8))
    # dataset='UCI'
    # y_values = [0.9406, 0.9583, 0.9476, 0.9600, 0.9593,0.9590]
    # y_values2=[0.9187,0.9361,0.9207,0.9108,0.9332,0.9493]
    # y_values3 = [0.8781, 0.9028, 0.9001, 0.895, 0.9172,0.9199]
    # y_values4=[0.9345,0.9301,0.9332,0.9359,0.9367,0.9332]
    # y_values5 = [0.9466, 0.9535, 0.9538, 0.9538, 0.9541,0.9525]
    # y_values6=[0.9365,	0.9381,	0.9346,	0.895,	0.895,	0.9005]

    # dataset = 'AS733'
    # y_values = [0.758,	0.7578	,0.7734,0.8145,0.8349,0.8729]
    # y_values2 = [0.7243,	0.9082,	0.8783,	0.9127	,0.9159,	0.9544]
    # y_values3 = [0.8787,	0.8749,	0.8983,	0.9151,	0.9183,	0.905]
    # y_values4 = [0.7578,	0.7152,	0.7344,	0.7793,	0.7907,	0.8124]
    # y_values5 = [0.7949,	0.8002,	0.8328	,0.88	,0.8989	,0.9362]
    # y_values6 = [0.9235,	0.8919,	0.941	,0.9473,	0.952,	0.9587]

    # dataset = 'Enron'
    # y_values = [0.6828,	0.7360	,0.8347,	0.8552,	0.8679,	0.8568]
    # y_values2 = [0.7179,	0.7490,	0.8444,	0.8667,	0.8803,	0.8687]
    # y_values3 = [0.6472,	0.7015,	0.7906,	0.8173,	0.8366,	0.8273]
    # y_values4 = [0.6986	,0.7211	,0.7148	,0.7975,	0.7362,	0.8001]
    # y_values5 = [0.7347,	0.7803,	0.8498,	0.8754,	0.8850,	0.8903]
    # y_values6 = [0.735	,0.8483	,0.9038,	0.8483,	0.8508,	0.8966]
    # 创建折线图
    plt.plot(x_values, y_values, marker='o', linestyle='-',label='EvolveGCN-H+STCDG')
    plt.plot(x_values, y_values2, marker='s', linestyle='-',label='EvolveGCN-O+STCDG' )
    plt.plot(x_values, y_values3, marker='*', linestyle='-', label='HTGN+STCDG')
    plt.plot(x_values, y_values4, marker='v', linestyle='-', label='SpikeNet+STCDG')
    plt.plot(x_values, y_values5, marker='x', linestyle='-', label='WinGNN+STCDG')
    plt.plot(x_values, y_values6, marker=',', linestyle='-', label='RTGCN+STCDG')
    # 添加标题和标签
    plt.title(dataset)
    plt.xlabel('The ratio of selected over-squashed nodes',fontsize=15)
    #plt.ylabel('AUC')
    plt.grid(True)

    #plt.text(0.5, 0.75, '{}'.format(dataset), fontsize=14, ha='center')
    # legend=plt.legend(ncol=6,loc='upper center', bbox_to_anchor=(0.5, 1.15))
    # for text in legend.get_texts():
    #     text.set_fontsize(20)  # 调整字体大小

    # 调整图例标记的大小
    # for handle in legend.legendHandles:
    #     handle.set_markersize(15)
    #     # 显示图形
    plt.show()
#plotParameter()
plot_embedding()
#plot_gradient_bar_and_get_color()
#get_spatial_fig_Compensate()
#get_spatial_fig_noCompensate()
#get_time_fig()
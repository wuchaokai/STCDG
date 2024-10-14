from __future__ import division, print_function
from sklearn.metrics import roc_auc_score,average_precision_score, recall_score, f1_score
import numpy as np
from sklearn import linear_model
from collections import defaultdict
import random

np.random.seed(123)
operatorTypes = ["HAD"]


def write_to_csv(test_results, output_name, model_name, dataset, time_steps, mod='val'):
    """Output result scores to a csv file for result logging"""
    with open(output_name, 'a+') as f:
        for op in test_results:
            print("{} results ({})".format(model_name, mod), test_results[op])
            _, best_auc = test_results[op]
            f.write("{},{},{},{},{},{},{}\n".format(dataset, time_steps, model_name, op, mod, "AUC", best_auc))


def get_link_score(fu, fv):
    """Given a pair of embeddings, compute link feature based on operator (such as Hadammad product, etc.)"""
    fu = np.array(fu)
    fv = np.array(fv)

    return np.multiply(fu, fv)



def get_link_feats(links, source_embeddings, target_embeddings):
    """Compute link features for a list of pairs"""
    features = []
    for l in links:
        a, b = l[0], l[1]
        f = get_link_score(source_embeddings[a], target_embeddings[b])
        features.append(f)
    return features


def get_random_split(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg):
    """ Randomly split a given set of train, val and test examples"""
    all_data_pos = []
    all_data_neg = []

    all_data_pos.extend(train_pos)
    all_data_neg.extend(train_neg)
    all_data_pos.extend(test_pos)
    all_data_neg.extend(test_neg)

    # re-define train_pos, train_neg, test_pos, test_neg.
    random.shuffle(all_data_pos)
    random.shuffle(all_data_neg)

    train_pos = all_data_pos[:int(0.2 * len(all_data_pos))]
    train_neg = all_data_neg[:int(0.2 * len(all_data_neg))]

    test_pos = all_data_pos[int(0.2 * len(all_data_pos)):]
    test_neg = all_data_neg[int(0.2 * len(all_data_neg)):]
    print("# train :", len(train_pos) + len(train_neg), "# val :", len(val_pos) + len(val_neg),
          "#test :", len(test_pos) + len(test_neg))
    return train_pos, train_neg, val_pos, val_neg, test_pos, test_neg


def evaluate_classifier(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, source_embeds, target_embeds):
    """Downstream logistic regression classifier to evaluate link prediction"""
    test_results = defaultdict(lambda: [])
    val_results = defaultdict(lambda: [])

    test_auc = get_roc_score_t(test_pos, test_neg, source_embeds, target_embeds)
    val_auc = get_roc_score_t(val_pos, val_neg, source_embeds, target_embeds)

    # Compute AUC based on sigmoid(u^T v) without classifier training.
    test_results['SIGMOID'].extend([test_auc, test_auc])
    val_results['SIGMOID'].extend([val_auc, val_auc])

    test_pred_true = defaultdict(lambda: [])
    val_pred_true = defaultdict(lambda: [])


    train_pos_feats = np.array(get_link_feats(train_pos, source_embeds, target_embeds))
    train_neg_feats = np.array(get_link_feats(train_neg, source_embeds, target_embeds))
    val_pos_feats = np.array(get_link_feats(val_pos, source_embeds, target_embeds))
    val_neg_feats = np.array(get_link_feats(val_neg, source_embeds, target_embeds))
    test_pos_feats = np.array(get_link_feats(test_pos, source_embeds, target_embeds))
    test_neg_feats = np.array(get_link_feats(test_neg, source_embeds, target_embeds))

    train_pos_labels = np.array([1] * len(train_pos_feats))
    train_neg_labels = np.array([0] * len(train_neg_feats))
    val_pos_labels = np.array([1] * len(val_pos_feats))
    val_neg_labels = np.array([0] * len(val_neg_feats))

    test_pos_labels = np.array([1] * len(test_pos_feats))
    test_neg_labels = np.array([0] * len(test_neg_feats))
    train_data = np.vstack((train_pos_feats, train_neg_feats))
    train_labels = np.append(train_pos_labels, train_neg_labels)

    val_data = np.vstack((val_pos_feats, val_neg_feats))
    val_labels = np.append(val_pos_labels, val_neg_labels)

    test_data = np.vstack((test_pos_feats, test_neg_feats))
    test_labels = np.append(test_pos_labels, test_neg_labels)

    logistic = linear_model.LogisticRegression()
    logistic.fit(train_data, train_labels)
    test_predict = logistic.predict_proba(test_data)[:, 1]
    val_predict = logistic.predict_proba(val_data)[:, 1]

    test_roc_score = roc_auc_score(test_labels, test_predict)
    val_roc_score = roc_auc_score(val_labels, val_predict)
    test_ap=average_precision_score(test_labels,test_predict)
    test_mrr=calculate_mrr(test_labels, test_predict)
    binary_pred=[1 if score >= 0.5 else 0 for score in min_max_scaling(test_predict)]

    test_recall=recall_score(test_labels,binary_pred)
    test_f1=f1_score(test_labels,binary_pred)
    #print(val_pos[35])
    #print(val_predict[35])
    val_test_predict=np.concatenate((val_predict, test_predict))
    val_test_binary_pred=[1 if score >= 0.5 else 0 for score in min_max_scaling(val_test_predict)]
    #print(val_test_binary_pred[35])
    val_test_label=np.concatenate((val_labels, test_labels))
    val_test_result=[1 if a == b else 0 for a, b in zip(val_test_binary_pred, val_test_label)]


    return val_roc_score, test_roc_score, test_ap, test_mrr,test_recall,test_f1,val_test_result,min_max_scaling(test_predict)[30]


def min_max_scaling(arr):
    """
    对 NumPy 数组进行最小-最大缩放。

    参数:
    arr: numpy.ndarray，输入数组

    返回:
    numpy.ndarray，缩放后的数组
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    scaled_arr = (arr - min_val) / (max_val - min_val)
    return scaled_arr
def evaluate_classifier2(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, source_embeds, target_embeds):
    """Downstream logistic regression classifier to evaluate link prediction"""
    test_results = defaultdict(lambda: [])
    val_results = defaultdict(lambda: [])

    test_auc = get_roc_score_t(test_pos, test_neg, source_embeds, target_embeds)
    val_auc = get_roc_score_t(val_pos, val_neg, source_embeds, target_embeds)

    # Compute AUC based on sigmoid(u^T v) without classifier training.
    test_results['SIGMOID'].extend([test_auc, test_auc])
    val_results['SIGMOID'].extend([val_auc, val_auc])

    test_pred_true = defaultdict(lambda: [])
    val_pred_true = defaultdict(lambda: [])


    train_pos_feats = np.array(get_link_feats(train_pos, source_embeds, target_embeds))
    train_neg_feats = np.array(get_link_feats(train_neg, source_embeds, target_embeds))
    val_pos_feats = np.array(get_link_feats(val_pos, source_embeds, target_embeds))
    val_neg_feats = np.array(get_link_feats(val_neg, source_embeds, target_embeds))
    test_pos_feats = np.array(get_link_feats(test_pos, source_embeds, target_embeds))
    test_neg_feats = np.array(get_link_feats(test_neg, source_embeds, target_embeds))


    train_pos_labels = np.array([1] * len(train_pos_feats))
    train_neg_labels = np.array([-1] * len(train_neg_feats))
    val_pos_labels = np.array([1] * len(val_pos_feats))
    val_neg_labels = np.array([-1] * len(val_neg_feats))

    test_pos_labels = np.array([1] * len(test_pos_feats))
    test_neg_labels = np.array([-1] * len(test_neg_feats))
    train_data = np.vstack((train_pos_feats, train_neg_feats))
    train_labels = np.append(train_pos_labels, train_neg_labels)

    val_data = np.vstack((val_pos_feats, val_neg_feats))
    val_labels = np.append(val_pos_labels, val_neg_labels)

    test_data = np.vstack((test_pos_feats, test_neg_feats))
    test_labels = np.append(test_pos_labels, test_neg_labels)

    logistic = linear_model.LogisticRegression()
    logistic.fit(train_data, train_labels)
    test_predict = logistic.predict_proba(test_data)[:, 1]
    val_predict = logistic.predict_proba(val_data)[:, 1]

    test_roc_score = roc_auc_score(test_labels, test_predict)
    val_roc_score = roc_auc_score(val_labels, val_predict)
    test_ap=average_precision_score(test_labels,test_predict)
    test_mrr=calculate_mrr(test_labels, test_predict)
    binary_pred=[1 if score >= 0.5 else -1 for score in test_predict]
    test_recall=recall_score(test_labels,binary_pred)
    test_f1=f1_score(test_labels,binary_pred)


    return val_roc_score, test_roc_score, test_ap, test_mrr,test_recall,test_f1

def calculate_mrr(test_labels, test_predict):
    # 获取正样例和负样例的索引
    positive_indices = np.where(test_labels == 1)[0]
    negative_indices = np.where(test_labels == 0)[0]

    rr_sum = 0.0
    for pos_idx in positive_indices:
        # 正样例 (u, v) 的分数
        pos_score = test_predict[pos_idx]

        # 生成候选链接集合 (包含该正样例及若干负样例)
        candidates_indices = [pos_idx] + list(
            np.random.choice(negative_indices, size=min(10, len(negative_indices)), replace=False))

        # 获取候选链接的分数
        candidate_scores = test_predict[candidates_indices]

        # 根据分数对候选链接进行排序
        sorted_indices = np.argsort(candidate_scores)[::-1]

        # 找到正样例在排序列表中的位置
        rank = np.where(sorted_indices == 0)[0][0] + 1

        # 计算 Reciprocal Rank
        rr_sum += 1.0 / rank

    # 计算 MRR
    mrr = rr_sum / len(positive_indices)
    return mrr

def get_roc_score_t(edges_pos, edges_neg, source_emb, target_emb):
    """Given test examples, edges_pos: +ve edges, edges_neg: -ve edges, return ROC scores for a given snapshot"""
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(source_emb, target_emb.T)
    pred = []
    pos = []
    for e in edges_pos:
        pred.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(1.0)

    pred_neg = []
    neg = []
    for e in edges_neg:
        pred_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(0.0)

    pred_all = np.hstack([pred, pred_neg])
    labels_all = np.hstack([np.ones(len(pred)), np.zeros(len(pred_neg))])
    roc_score = roc_auc_score(labels_all, pred_all)
    return roc_score
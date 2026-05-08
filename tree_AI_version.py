import pandas as pd
import numpy as np
from collections import Counter

# 假设 LABEL_COL 是全局变量，这里定义为 'revenue'
LABEL_COL = 'revenue'

def calc_entropy(data, label=LABEL_COL):
    """
    计算数据集在标签列上的熵
    :param data: DataFrame，包含标签列
    :param label: 标签列名
    :return: 熵值 (float)
    """
    # 统计各类别样本数
    counts = data[label].value_counts()
    total = len(data)
    # 计算熵
    entropy = 0.0
    for count in counts:
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy


def calc_info_gain(data, feature, label=LABEL_COL):
    """
    计算某个特征的信息增益
    :param data: DataFrame
    :param feature: 特征列名
    :param label: 标签列名
    :return: 信息增益 (float)
    """
    # 1. 计算原始熵 H(Y)
    H_Y = calc_entropy(data, label)

    # 2. 计算条件熵 H(Y|X)
    # 按特征取值分组，每组计算熵，然后加权平均
    grouped = data.groupby(feature)
    H_Y_given_X = 0.0
    total = len(data)
    for value, group in grouped:
        weight = len(group) / total
        H_Y_given_X += weight * calc_entropy(group, label)

    # 信息增益 = H(Y) - H(Y|X)
    info_gain = H_Y - H_Y_given_X
    return info_gain


def majority_class(data, label=LABEL_COL):
    """
    返回当前数据集中出现次数最多的类别
    :param data: DataFrame
    :param label: 标签列名
    :return: 类别值 (例如 'High', 'Low' 或 0,1,2 等)
    """
    return data[label].mode()[0]


def build_tree(data, features, label=LABEL_COL, max_depth=None, current_depth=0):
    """
    递归构建多叉 ID3 决策树
    :param data: 当前节点的数据集
    :param features: 可用的特征列表（排除已使用的）
    :param label: 标签列名
    :param max_depth: 树的最大深度（可选，用于预剪枝）
    :param current_depth: 当前深度（递归内部使用）
    :return: 决策树字典
    """
    # 计算当前节点的多数类（用于 fallback）
    majority = majority_class(data, label)

    # 停止条件1：当前深度达到最大深度（预剪枝选做）
    if max_depth is not None and current_depth >= max_depth:
        return majority

    # 停止条件2：所有样本属于同一类别 -> 叶节点
    if data[label].nunique() == 1:
        return data[label].iloc[0]

    # 停止条件3：没有可用特征 -> 返回多数类
    if not features:
        return majority

    # 选择信息增益最大的特征
    best_feature = None
    best_gain = -1
    for feature in features:
        gain = calc_info_gain(data, feature, label)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    # 如果最佳信息增益为0（特征无区分度），直接返回多数类
    if best_gain == 0:
        return majority

    # 构建内部节点
    tree = {
        'feature': best_feature,
        'majority': majority,          # 存储多数类，用于预测时 fallback
        'children': {}
    }

    # 获取该特征在数据中所有可能的取值
    values = data[best_feature].unique()
    # 从可用特征列表中移除当前特征（ID3 通常不重复使用特征）
    remaining_features = [f for f in features if f != best_feature]

    for value in values:
        # 划分出该特征取值对应的子集
        subset = data[data[best_feature] == value]
        if len(subset) == 0:
            # 按理说不会出现，但安全起见使用多数类
            subtree = majority
        else:
            # 递归构建子树
            subtree = build_tree(subset, remaining_features, label,
                                 max_depth, current_depth + 1)
        tree['children'][value] = subtree

    return tree


def predict_one(tree, sample):
    """
    对单个样本进行预测
    :param tree: 决策树（字典或叶节点值）
    :param sample: 单个样本（pandas Series 或 dict，包含特征值）
    :return: 预测类别
    """
    # 如果是叶节点（类别值），直接返回
    if not isinstance(tree, dict):
        return tree

    # 内部节点：获取特征名
    feature = tree['feature']
    value = sample.get(feature)  # 获取样本在该特征上的值

    # 如果特征值在训练中未出现过，或不存在，则 fallback 到 majority
    if value not in tree['children']:
        return tree['majority']

    # 递归进入对应子节点
    return predict_one(tree['children'][value], sample)


def predict(tree, data):
    """
    对数据集进行批量预测
    :param tree: 决策树
    :param data: DataFrame，包含特征列
    :return: 预测结果列表（与数据行顺序一致）
    """
    predictions = []
    for idx, row in data.iterrows():
        predictions.append(predict_one(tree, row))
    return predictions
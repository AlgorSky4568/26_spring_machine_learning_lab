LABEL_COL = 'revenue'
import math
def calc_entropy(data, label=LABEL_COL):
    rows = data.shape[0]
    if rows == 0: return 0.0
    count = data[label].value_counts()
    entropy = 0.0
    for cnt in count:
        p = cnt / rows
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def calc_info_gain(data, feature, label=LABEL_COL):
    rows = data.shape[0]
    count = data[feature].value_counts()
    entropy_feature = 0.0
    for value, cnt in count.items():
        sub_data = data[data[feature] == value]
        entropy_feature += (cnt / rows) * calc_entropy(sub_data, label=label)
    return calc_entropy(data, label=label) - entropy_feature

def majority_class(data, label=LABEL_COL):
    return data[label].mode()[0]
    #返回剩下样本中最多的类别，在这里只有0和1两个类别，那就看1最多还是0最多，少数服从多数

def build_tree(data, features, label=LABEL_COL, max_depth=None, current_depth=0):
    majority = majority_class(data,label) #当前数据集中的多数，叶子节点进行不下去或者遇到新的属性值判断不了时，就用majority
    if max_depth is not None and current_depth >= max_depth: #表示到了叶子节点
        return majority

    if data[label].nunique() == 1:
        return data[label].iloc[0]
    if not features:
        return majority


    best_feature = features[0]
    gain = 0.0
    for feature in features:
        info_gain = calc_info_gain(data, feature, label)
        if info_gain > gain:
            gain = info_gain
            best_feature = feature

    if gain == 0:
        return majority

    tree = {
        'feature': best_feature,
        'majority': majority,
        'children':{}
    }
    values = data[best_feature].unique()
    remaining_feature = [f for f in features if f != best_feature]

    for value in values:
        subset = data[data[best_feature] == value]
        if len(subset) == 0:
            tree['children'][value] = majority
        elif len(subset) == 1: #表示D3算法不能再分了，就是叶子节点了
            pass
        else:
            tree['children'][value] = build_tree(subset,remaining_feature,label,max_depth,current_depth+1)

    return tree

def predict_one(tree, sample):
    if not isinstance(tree, dict):
        return tree
    else:
        feature = tree['feature']
        value = sample.get(feature)
        if value not in tree['children']:
            return tree['majority']

        return predict_one(tree['children'][value],sample)

def predict(tree,data):
    predictions = []
    for index, row in data.iterrows():
        predictions.append(predict_one(tree,row))
    return predictions
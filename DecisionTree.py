# -*- coding: utf-8 -*-
# load module
import time
from sklearn.tree import DecisionTreeClassifier
from LoadDataset import loadDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

def DecisionTree(normal_filename, normal_cnt, attack_filename, attack_cnt):
    df = loadDataset(normal_filename, normal_cnt, attack_filename, attack_cnt)
    data = df.iloc[:, 5:-1]
    label = df.iloc[:, -1:]
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3,
                                                        random_state=42)

    # 决策树
    clf = DecisionTreeClassifier(class_weight='balanced', random_state=37)
    clf = clf.fit(x_train, y_train)  # 拟合训练集
    score_c = clf.score(x_test, y_test)  # 输出测试集准确率

    print('DecisionTree score: ', score_c)

    # 决策树 预测测试集
    y_test_proba_clf = clf.predict_proba(x_test)
    
    y_pred =clf.predict(x_test)
    clf_accuracy = accuracy_score(y_test, y_pred)
    clf_precision = precision_score(y_test, y_pred)
    clf_recall = recall_score(y_test, y_pred)
    clf_f1 = f1_score(y_test, y_pred)
    print("decision_tree_accuracy : ",clf_accuracy)
    print("decision_tree_precision : ",clf_precision)
    print("decision_tree_recall : ",clf_recall)
    print("decision_tree_f1 : ",clf_f1)
    
    false_positive_rate_clf, recall_clf, thresholds_clf = roc_curve(y_test, y_test_proba_clf[:, 1])
    # 决策树 AUC指标
    roc_auc_clf = auc(false_positive_rate_clf, recall_clf)

    # 画图 画出模型的ROC曲线
    plt.plot(false_positive_rate_clf, recall_clf, label='DecisionTree_AUC=%0.4f' % roc_auc_clf)
    plt.legend(loc='best', fontsize=15, frameon=False)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.show()

    # 特征贡献度
    columns = data.columns
    FI = pd.Series(clf.feature_importances_, index=columns)  # sklearn
    FI = FI.sort_values(ascending=False)
    fig = plt.figure(figsize=(15, 10))
    plt.bar(FI.index, FI.values, width=0.5, bottom=None, color="blue")

    plt.yticks(fontproperties='Times New Roman', size=10)
    plt.xticks(fontproperties='Times New Roman', size=10, rotation=90)

    plt.ylabel('features', fontsize=10)
    plt.xlabel('importances', fontsize=10)
    plt.show()

    print("特征贡献度：" + str(FI.index))


if __name__ == "__main__":
    DecisionTree("./normal.csv", 25587, "./mal.csv", 1104)

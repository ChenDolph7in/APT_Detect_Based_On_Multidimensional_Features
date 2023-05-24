# -*- coding: utf-8 -*-
# load module
import time
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from LoadDataset import loadDataset
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt


def XGBoost(normal_filename, normal_cnt, attack_filename, attack_cnt):
    df = loadDataset(normal_filename, normal_cnt, attack_filename, attack_cnt)
    data = df.iloc[:, 5:-1]
    label = df.iloc[:, -1:]
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3,
                                                        random_state=42)

    # fit model for train data
    # 建立XgBoost模型
    xgb_class_model = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,  # 树的个数--1000棵树建立xgboost
        max_depth=6,  # 树的深度
        min_child_weight=1,  # 叶子节点最小权重
        gamma=0.,  # 惩罚项中叶子结点个数前的参数
        # subsample=0.8,  # 随机选择80%样本建立决策树
        # colsample_btree=1,  # 随机选择80%特征建立决策树
        objective='binary:logitraw',  # 指定目标函数，二分类
        scale_pos_weight=1,  # 解决样本个数不平衡的问题
        random_state=27  # 随机数
    )
    # 训练模型
    # print(x_train)
    xgb_class_model.fit(x_train,
                        y_train,
                        eval_set=[(x_test, y_test)],
                        eval_metric="error",
                        early_stopping_rounds=10,
                        verbose=True
                        )

    # make prediction for test data
    y_pred = xgb_class_model.predict(x_test)
    # 这里是直接给出类型，predict_proba()函数是给出属于每个类别的概率。

    accuracy = accuracy_score(y_test, y_pred)
    print("accuarcy: %.2f%%" % (accuracy * 100.0))

    # XgBoost 预测测试集
    y_pred_proba = xgb_class_model.predict_proba(x_test)

    
    xgb_accuracy = accuracy_score(y_test, y_pred)
    xgb_precision = precision_score(y_test, y_pred)
    xgb_recall = recall_score(y_test, y_pred)
    xgb_f1 = f1_score(y_test, y_pred)
    print("xgboost_accuracy : ", xgb_accuracy)
    print("xgboost_precision : ", xgb_precision)
    print("xgboost_recall : ", xgb_recall)
    print("xgboost_f1 : ", xgb_f1)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
    roc_auc_xgboost = auc(fpr, tpr)
    # plt.figure()
    # lw = 2
    # plt.plot(fpr,tpr,color="darkorange",lw=lw,label="XgBoost_AUC=%0.4f" % roc_auc_xgboost,)
    # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.legend(loc="best")
    # # plt.savefig('auc_roc.pdf')
    # plt.show()

    # 画图 画出模型的ROC曲线
    plt.plot(fpr,tpr, label='DecisionTree_AUC=%0.4f' % roc_auc_xgboost)
    plt.legend(loc='best', fontsize=15, frameon=False)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.show()

    # 特征贡献度
    fig, ax = plt.subplots(figsize=(15, 9))
    plt.rcParams.update({'font.size': 24})
    plt.tick_params(labelsize=24)
    plt.legend(loc='upper right')
    ax.set_xlabel(..., fontsize=24)
    ax.set_ylabel(..., fontsize=24)
    plot_importance(xgb_class_model,
                    height=0.7,
                    ax=ax,
                    importance_type='gain',
                    max_num_features=20
                    )
    plt.show()

    # 特征贡献度
    columns = data.columns
    FI = pd.Series(xgb_class_model.feature_importances_, index=columns)  # sklearn
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
    XGBoost("./normal.csv", 25587, "./mal.csv", 1104)

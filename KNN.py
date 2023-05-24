# -*- coding: utf-8 -*-
# load module
import time
from sklearn.neighbors import KNeighborsClassifier
from LoadDataset import loadDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

def KNN(normal_filename, normal_cnt, attack_filename, attack_cnt):
    df = loadDataset(normal_filename, normal_cnt, attack_filename, attack_cnt)
    data = df.iloc[:, 5:-1]
    label = df.iloc[:, -1:]
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3,
                                                        random_state=42)

    # 建立knn分类模型，并指定 k 值
    knc = KNeighborsClassifier(n_neighbors=10)
    # 使用训练集训练模型
    start_time = time.time()
    knc.fit(x_train,y_train)
    end_time = time.time()
    print("花费时间为{}".format(end_time - start_time))

    score=knc.score(x_test,y_test)
    print('KNN score: ', score)

    # KNN 预测测试集
    y_test_proba_knn = knc.predict_proba(x_test)

    y_pred = knc.predict(x_test)
    knn_accuracy = accuracy_score(y_test, y_pred)
    knn_precision = precision_score(y_test, y_pred)
    knn_recall = recall_score(y_test, y_pred)
    knn_f1 = f1_score(y_test, y_pred)
    print("knn_accuracy : ",knn_accuracy)
    print("knn_precision : ",knn_precision)
    print("knn_recall : ", knn_recall)
    print("knn_f1 : ", knn_f1)

    false_positive_rate_knn, recall_knn, thresholds_knn = roc_curve(y_test, y_test_proba_knn[:, 1])
    # KNN AUC指标
    roc_auc_knn = auc(false_positive_rate_knn, recall_knn)

    # 画图 画出模型对应的ROC曲线
    plt.plot(false_positive_rate_knn, recall_knn, label='knn_AUC=%0.4f' % roc_auc_knn)
    plt.legend(loc='best', fontsize=15, frameon=False)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.show()

    # # 特征贡献度
    # columns = data.columns
    # FI = pd.Series(knc.feature_importances_, index=columns)  # sklearn
    # FI = FI.sort_values(ascending=False)
    # fig = plt.figure(figsize=(15, 10))
    # plt.bar(FI.index, FI.values, width=0.5, bottom=None, color="blue")
    #
    # plt.yticks(fontproperties='Times New Roman', size=10)
    # plt.xticks(fontproperties='Times New Roman', size=10, rotation=90)
    #
    # plt.ylabel('features', fontsize=10)
    # plt.xlabel('importances', fontsize=10)
    # plt.show()
    #
    # print("特征贡献度：" + str(FI.index))

if __name__ == "__main__":
    KNN("./normal.csv", 25587, "./mal.csv", 1104)
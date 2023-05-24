# -*- coding: utf-8 -*-
# load module
import time
from sklearn.ensemble import RandomForestClassifier
from LoadDataset import loadDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt


def RandomForest(normal_filename, normal_cnt, attack_filename, attack_cnt):
    df = loadDataset(normal_filename, normal_cnt, attack_filename, attack_cnt)
    data = df.iloc[:, 5:-1]
    label = df.iloc[:, -1:]
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3,
                                                        random_state=42)

    # 随机森林
    rfc = RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=37)
    rfc = rfc.fit(x_train, y_train)
    score_r = rfc.score(x_test, y_test)

    print('RandomForest score: ', score_r)

    # 随机森林 预测测试集
    y_test_proba_rfc = rfc.predict_proba(x_test)

    y_pred = rfc.predict(x_test)
    rfc_accuracy = accuracy_score(y_test, y_pred)
    rfc_precision = precision_score(y_test, y_pred)
    rfc_recall = recall_score(y_test, y_pred)
    rfc_f1 = f1_score(y_test, y_pred)
    print("randomforest_accuracy : ", rfc_accuracy)
    print("randomforest_precision : ", rfc_precision)
    print("randomforest_recall : ", rfc_recall)
    print("randomforest_f1 : ", rfc_f1)

    false_positive_rate_rfc, recall_rfc, thresholds_rfc = roc_curve(y_test, y_test_proba_rfc[:, 1])
    # 随机森林 AUC指标
    roc_auc_rfc = auc(false_positive_rate_rfc, recall_rfc)

    # 画图 画出模型的ROC曲线
    plt.plot(false_positive_rate_rfc, recall_rfc, label='RandomForest_AUC=%0.4f' % roc_auc_rfc)
    plt.legend(loc='best', fontsize=15, frameon=False)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.show()

    # 特征贡献度
    columns = data.columns
    FI = pd.Series(rfc.feature_importances_, index=columns)  # sklearn
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
    RandomForest("./normal.csv", 25587, "./mal.csv", 1104)

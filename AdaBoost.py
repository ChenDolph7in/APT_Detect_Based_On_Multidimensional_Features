# -*- coding: utf-8 -*-
# load module
import time
from sklearn.ensemble import AdaBoostClassifier
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

    # 建立AdaBoost模型
    abc = AdaBoostClassifier(n_estimators=100, random_state=37)
    abc = abc.fit(x_train, y_train)
    score_r = abc.score(x_test, y_test)

    print('AdaBoost score: ', score_r)

    # 随机森林 预测测试集
    y_test_proba_abc = abc.predict_proba(x_test)

    y_pred = abc.predict(x_test)
    abc_accuracy = accuracy_score(y_test, y_pred)
    abc_precision = precision_score(y_test, y_pred)
    abc_recall = recall_score(y_test, y_pred)
    abc_f1 = f1_score(y_test, y_pred)
    print("abc_accuracy : ", abc_accuracy)
    print("abc_precision : ", abc_precision)
    print("abc_recall : ", abc_recall)
    print("abc_f1 : ", abc_f1)
    
    false_positive_rate_abc, recall_abc, thresholds_abc = roc_curve(y_test, y_test_proba_abc[:, 1])
    # 随机森林 AUC指标
    roc_auc_abc = auc(false_positive_rate_abc, recall_abc)

    # 画图 画出俩模型的ROC曲线
    plt.plot(false_positive_rate_abc, recall_abc, label='AdaBoost_AUC=%0.4f' % roc_auc_abc)
    plt.legend(loc='best', fontsize=15, frameon=False)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.show()

    # 特征贡献度
    columns = data.columns
    FI = pd.Series(abc.feature_importances_, index=columns)  # sklearn
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

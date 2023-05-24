import numpy as np
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from LoadDataset import loadDataset
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

from matplotlib import pyplot as plt

def GBDT(normal_filename, normal_cnt, attack_filename, attack_cnt):
    df = loadDataset(normal_filename, normal_cnt, attack_filename, attack_cnt)
    data = df.iloc[:, 5:-1]
    label = df.iloc[:, -1:]
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3,
                                                        random_state=42)

    y_test = y_test.values.ravel()
    y_train = y_train.values.ravel()

    # 建立XgBoost模型
    gbdt = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=5, subsample=1
                                      , min_samples_split=2, min_samples_leaf=1, max_depth=3
                                      , init=None, random_state=None, max_features=None
                                      , verbose=0, max_leaf_nodes=None, warm_start=False
                                      )

    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    gbdt.fit(x_train, y_train)

    score = gbdt.score(x_test, y_test)
    print("GBDT score: ", score)
    # pred = gbdt.predict(x_test)
    # total_err = 0
    # for i in range(pred.shape[0]):
    #     print(pred[i], y_test[i])
    #     err = (pred[i] - y_test[i]) / y_test[i]
    #     total_err += err * err
    # print(total_err / pred.shape[0])

    # GBDT 预测测试集
    y_test_proba_gbdt = gbdt.predict_proba(x_test)

    y_pred = gbdt.predict(x_test)
    gbdt_accuracy = accuracy_score(y_test, y_pred)
    gbdt_precision = precision_score(y_test, y_pred)
    gbdt_recall = recall_score(y_test, y_pred)
    gbdt_f1 = f1_score(y_test, y_pred)
    print("gbdt_accuracy : ", gbdt_accuracy)
    print("gbdt_precision : ", gbdt_precision)
    print("gbdt_recall : ", gbdt_recall)
    print("gbdt_f1 : ", gbdt_f1)
    
    false_positive_rate_gbdt, recall_gbdt, thresholds_gbdt = roc_curve(y_test, y_test_proba_gbdt[:, 1])
    # GBDT AUC指标
    roc_auc_gbdt = auc(false_positive_rate_gbdt, recall_gbdt)

    # 画图 画出模型对应的ROC曲线
    plt.plot(false_positive_rate_gbdt, recall_gbdt, label='GBDT_AUC=%0.4f' % roc_auc_gbdt)
    plt.legend(loc='best', fontsize=15, frameon=False)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.show()

    # 特征贡献度
    columns = data.columns
    FI = pd.Series(gbdt.feature_importances_, index=columns)  # sklearn
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
    GBDT("./normal.csv", 25587, "./mal.csv", 1104)

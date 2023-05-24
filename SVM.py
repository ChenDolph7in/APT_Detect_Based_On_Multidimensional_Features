# -*- coding: utf-8 -*-
# load module
import time

from sklearn import svm
from sklearn.model_selection import train_test_split
from LoadDataset import loadDataset
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

def SVM(normal_filename, normal_cnt, attack_filename, attack_cnt):
    df = loadDataset(normal_filename, normal_cnt, attack_filename, attack_cnt)
    data = df.iloc[:, 5:-1]
    label = df.iloc[:, -1:]
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3,
                                                        random_state=42)


    print(data)
    # print(label)
    clf = svm.SVC(kernel='poly',probability=True)
    start_time = time.time()
    clf.fit(x_train,y_train)
    end_time = time.time()
    print("花费时间为{}".format(end_time - start_time))

    score=clf.score(x_test,y_test)
    print('svm score: ', score)

    # svm 预测测试集
    y_pred = clf.predict(x_test)
    svm_accuracy = accuracy_score(y_test, y_pred)
    svm_precision = precision_score(y_test, y_pred)
    svm_recall = recall_score(y_test, y_pred)
    svm_f1 = f1_score(y_test, y_pred)
    print("svm_accuracy : ",svm_accuracy)
    print("svm_precision : ",svm_precision)
    print("svm_recall : ", svm_recall)
    print("svm_f1 : ", svm_f1)

    y_test_proba_svm = clf.predict_proba(x_test)
    
    false_positive_rate_svm, recall_svm, thresholds_svm = roc_curve(y_test, y_test_proba_svm[:, 1])
    # SVM AUC指标
    roc_auc_svm = auc(false_positive_rate_svm, recall_svm)

    # 画图 画出模型对应的ROC曲线
    plt.plot(false_positive_rate_svm, recall_svm, label='svm_AUC=%0.4f' % roc_auc_svm)
    plt.legend(loc='best', fontsize=15, frameon=False)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.show()


if __name__ == "__main__":
    SVM("./normal.csv", 19000, "./mal.csv", 1000)
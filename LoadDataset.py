# -*- coding: utf-8 -*-
import pandas as pd

def loadDataset(normal_filename, normal_cnt, attack_filename, attack_cnt):
    normal_dataset = pd.read_csv(normal_filename, encoding='utf-8', index_col=0).iloc[0:normal_cnt, :]
    normal_dataset['label'] = [0 for i in range(normal_cnt)]
    attack_dataset = pd.read_csv(attack_filename, encoding='utf-8', index_col=0).iloc[0:attack_cnt, :]
    attack_dataset['label'] = [1 for i in range(attack_cnt)]

    df = pd.concat([normal_dataset, attack_dataset])
    index_conn_state(df) # 处理非数值特征

    # print(df)
    # print(df['conn_state'])
    return df.fillna(0) # 处理NAN特征

def index_conn_state(df):
    index_dict = {'S0': 0, 'S1': 1, 'SF': 2, 'REJ': 3, 'S2': 4, 'S3': 5, 'RSTO': 6, 'RSTR': 7, 'RSTOS0': 8, 'RSTRH': 9,
                  'SH': 10, 'SHR': 11, 'OTH': 12}

    df['conn_state'] = df.conn_state.apply(lambda x: index_dict[x] if x in index_dict.keys() else -1)


if __name__ == "__main__":
    df = loadDataset("./mal.csv", 1000, "./normal.csv", 19000)
    print(df)
    df1 = df.iloc[:,5:-1]
    print(df1)
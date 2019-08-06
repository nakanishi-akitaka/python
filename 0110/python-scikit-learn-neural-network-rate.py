# -*- coding: utf-8 -*-
"""
【Scikit-learn】ニューラルネットワークの識別率を計算
https://algorithm.joho.info/machine-learning/
    python-scikit-learn-neural-network-rate/

Created on Thu Jan 10 14:36:54 2019

@author: Akitaka
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier

def main():
    # データを取得
    data = pd.read_csv("data.csv", sep=",")

    # ニューラルネットで学習
    clf = MLPClassifier(solver="sgd",random_state=0,max_iter=10000)

    # 学習(説明変数x1, x2、目的変数x3)
    clf.fit(data[['x1', 'x2']], data['x3'])

    # 学習データを元に説明変数x1, x2から目的変数x3を予測
    pred = clf.predict(data[['x1', 'x2']])

    # 識別率を表示
    print (sum(pred == data['x3']) / len(data[['x1', 'x2']]))
    
if __name__ == "__main__":
    main()
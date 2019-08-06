# -*- coding: utf-8 -*-
"""
【Scikit-learn】ニューラルネットワーク学習モデルのファイル出力・保存
https://algorithm.joho.info/machine-learning/
    python-scikit-learn-neural-network-dump/

Created on Thu Jan 10 14:39:44 2019

@author: Akitaka
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

def main():
    # データを取得
    data = pd.read_csv("data.csv", sep=",")

    # ニューラルネットで学習
    clf = MLPClassifier(solver="sgd",random_state=0,max_iter=10000)

    # 学習(説明変数x1, x2、目的変数x3)
    clf.fit(data[['x1', 'x2']], data['x3'])

    # 学習データを元に説明変数x1, x2から目的変数x3を予測
    pred = clf.predict(data[['x1', 'x2']])

    # 結果表示
    print (pred)
    joblib.dump(clf, 'nn.learn')
    
if __name__ == "__main__":
    main()
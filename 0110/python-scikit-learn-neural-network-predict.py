# -*- coding: utf-8 -*-
"""
【Scikit-learn】ニューラルネットワーク（多層パーセプトロン・MLP）
https://algorithm.joho.info/machine-learning/
    python-scikit-learn-neural-network-predict/

Created on Thu Jan 10 14:32:00 2019

@author: Akitaka
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

def main():
    # 学習用のデータを読み込み
    data = pd.read_csv("train.csv", sep=",")

    # 説明変数：x1, x2
    X = data.loc[:, ['x1', 'x2']].values

    # 目的変数：x3
    y = data['x3'].values

    # 学習
    clf = MLPClassifier(solver="sgd",random_state=0,max_iter=10000)
    clf.fit(X, y)

    # テスト用データの読み込み
    data = pd.read_csv("test.csv", sep=",")

    # 学習結果の検証（テスト用データx1, x2を入力）
    X_test = data.loc[:, ['x1', 'x2']].values
    y_predict = clf.predict(X_test)

    # 検証結果の表示
    print("検証結果：", y_predict)

    # 学習結果を出力
    joblib.dump(clf, 'train.learn') 

if __name__ == "__main__":
    main()

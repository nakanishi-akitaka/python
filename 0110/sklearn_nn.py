# -*- coding: utf-8 -*-
"""
Python(scikit-learn)でニューラルネットワーク
https://qiita.com/tsal3290s/items/3c0b8713a26ee10b689e

Created on Thu Jan 10 13:10:54 2019

@author: Akitaka
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
clf = MLPClassifier(solver="sgd",random_state=0,max_iter=10000)
clf.fit(X_train, y_train)
print (clf.score(X_test, y_test))
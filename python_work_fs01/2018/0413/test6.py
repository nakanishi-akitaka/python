#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf = svm.SVC()
clf.fit(X, y)
import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print(clf2.predict(X[0:1]))
print(y[0])

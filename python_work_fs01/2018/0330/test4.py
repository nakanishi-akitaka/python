#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import datasets
iris = datasets.load_iris()
# print(iris['feature_names'])
# print(iris['target_names'])
# print(iris['data'])
# print(iris['target'])

from sklearn.model_selection import train_test_split
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_predicted = clf.predict(X_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_predicted)
print(accuracy)

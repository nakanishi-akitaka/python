#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm

iris = datasets.load_iris()
# print(iris.data)
# print(iris.target)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, random_state=0)
clf = svm.SVC()
clf.fit(X_train, y_train)
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
# decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
# max_iter=-1, probability=False, random_state=None, shrinking=True,
# tol=0.001, verbose=False)
print("pred values = ",list(clf.predict(X_test)))
print("true values =",list(y_test))
print("accyracy = ",clf.score(X_test, y_test))

from sklearn.decomposition import PCA
pca = PCA(1)
pca.fit_transform(X_train, y_train)
clfp = Pipeline([
('dim',PCA(1)),
('svm',svm.SVC())
])
clfp.fit(X_train, y_train)
print("pred values = ",list(clfp.predict(X_test)))
print("true values =",list(y_test))
print("accyracy = ",clfp.score(X_test, y_test))

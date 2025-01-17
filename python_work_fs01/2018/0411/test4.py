#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3.1
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

print(X_train.shape,y_train.shape)
print(X_test.shape, y_test.shape )
clf=svm.SVC(kernel='linear',C=1).fit(X_train,y_train)
print(clf.score(X_test,y_test))

# 3.1.1
from sklearn.model_selection import cross_val_score
clf=svm.SVC(kernel='linear',C=1)
scores=cross_val_score(clf, iris.data, iris.target, cv=5)
print(scores)
print("Accyracy: %0.2F (+/- %0.2f)" % (scores.mean(), scores.std()*2))

from sklearn import metrics
scores=cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro')
print(scores)

from sklearn.model_selection import ShuffleSplit
n_samples = iris.data.shape[0]
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
print(cross_val_score(clf, iris.data, iris.target, cv=cv))

from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
print(clf.score(X_test_transformed, y_test))

from sklearn.pipeline import make_pipeline
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
print(cross_val_score(clf, iris.data, iris.target, cv=cv))

# 3.1.1.1
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)
print(metrics.accuracy_score(iris.target, predicted))

# 3.1.2
# nothing

# 3.1.3
# nothing

# 3.1.3.1 KFold
from sklearn.model_selection import KFold
X = ["a","b","c","d"]
kf = KFold(n_splits=2)
for train, test in kf.split(X):
    print("%s %s" % (train, test))
X = np.array([[0., 0.], [1., 1.], [-1., -1.], [2., 2.]])
y = np.array([0, 1, 0, 1])
X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

# 3.1.3.2
from sklearn.model_selection import LeaveOneOut
X = ["1","2","3","4"]
loo = LeaveOneOut()
for train, test in loo.split(X):
    print("%s %s" % (train, test))

# 3.1.3.3 
from sklearn.model_selection import LeavePOut
X = np.ones(4)
lpo = LeavePOut(p=2)
for train, test in lpo.split(X):
    print("%s %s" % (train, test))

# 3.1.3.4
from sklearn.model_selection import ShuffleSplit
X = np.arange(5)
ss = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
for train_index, test_index in ss.split(X):
    print("%s %s" % (train_index, test_index))

# 3.1.4
# nothing

# 3.1.4.1
from sklearn.model_selection import StratifiedKFold
X = np.ones(10)
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
    print("%s %s" % (train, test))

# 3.1.5
# nothing

# 3.1.5.1
from sklearn.model_selection import GroupKFold
X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
gkf = GroupKFold(n_splits=3)
for train, test in gkf.split(X, y, groups=groups):
    print("%s %s" % (train, test))

# 3.1.5.2
from sklearn.model_selection import LeaveOneGroupOut
X = [1, 5, 10, 50, 60, 70, 80]
y = [0, 1, 1, 2, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3, 3]
logo = LeaveOneGroupOut()
for train, test in logo.split(X, y, groups=groups):
    print("%s %s" % (train, test))

# 3.1.5.3
from sklearn.model_selection import LeavePGroupsOut
X = np.arange(6)
y = [1, 1, 1, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3]
lpgo = LeavePGroupsOut(n_groups=2)
for train, test in lpgo.split(X, y, groups=groups):
    print("%s %s" % (train, test))

# 3.1.5.4
from sklearn.model_selection import GroupShuffleSplit
X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 0.001]
y = ["a", "b", "b", "b", "c", "c", "c", "a"]
groups = [1, 1, 2, 2, 3, 3, 4, 4]
gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
for train, test in gss.split(X, y, groups=groups):
    print("%s %s" % (train, test))

# 3.1.6
# nothing

# 3.1.7
# nothing

# 3.1.7.1
from sklearn.model_selection import TimeSeriesSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=3)
for train, test in tscv.split(X):
    print("%s %s" % (train, test))

# 3.1.8
# nothing

# 3.1.9
# nothing
# 3.1.9

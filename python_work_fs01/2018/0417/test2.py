#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 1.4.1.
from sklearn import svm
X = [[0,0],[1,1]]
y = [0,1]
clf = svm.SVC()
clf.fit(X, y)
y_pred = clf.predict([[2., 2.]])
print(y_pred)
print('support_vectors')
print(clf.support_vectors_)
print('indices of support vectors')
print(clf.support_)
print('number of support vectors for each class')
print(clf.n_support_)

# 1.4.1.1.
X = [[0], [1], [2], [3]]
y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, y)
dec = clf.decision_function([[1]])
print(dec.shape[1])
clf.decision_function_shape = 'ovr'
dec = clf.decision_function([[1]])
print(dec.shape[1])

lin_clf = svm.LinearSVC()
lin_clf.fit(X, y)
dec = lin_clf.decision_function([[1]])
print(dec.shape[1])

# 1.4.2.
from sklearn import svm
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = svm.SVR()
clf.fit(X, y)
y_pred = clf.predict([[1, 1]])
print(y_pred)



from scipy.stats import expon
print(float(expon(scale=0.01)))

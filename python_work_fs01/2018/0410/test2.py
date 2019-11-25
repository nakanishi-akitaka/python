#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import datasets
iris = datasets.load_iris()

from sklearn.model_selection import train_test_split
X = iris.data   # = iris['data']
y = iris.target # = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print('score=',clf.score(X_train,y_train))


from sklearn import metrics
y_predicted = clf.predict(X_test)
print('metrics.classification_report')
print( metrics.classification_report(y_test, y_predicted,target_names=iris.target_names))

prob=clf.predict_proba(X_test)[:,2]
# print(clf.predict_proba(X_test))
# print(clf.predict_proba(X_test)[:2])
# print(clf.predict_proba(X_test)[:,2])
fpr,tpr,thresholds=metrics.roc_curve(y_test,prob,pos_label=2)
print('auc=',metrics.auc(fpr,tpr))
# print(y_test)
# print(X_test)
# print(prob)
# print(fpr)
# print(tpr)
# from matplotlib import pyplot
# pyplot.plot(fpr,tpr)
# pyplot.title('ROC curve')
# pyplot.xlabel('False Positive Rate')
# pyplot.ylabel('True Positive Rate')
# pyplot.show()

#### http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
import numpy
y_true=numpy.array([0,0,1,1])
y_pred=numpy.array([0.1,0.4,0.35,0.8])
print('metrics.roc_auc_score(y_true,y_pred)')
print( metrics.roc_auc_score(y_true,y_pred))

#### http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
#### http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
import numpy
y_true=numpy.array([1,1,2,2])
y_pred=numpy.array([0.1,0.4,0.35,0.8])
fpr,tpr,thresholds=metrics.roc_curve(y_true,y_pred,pos_label=2)
print('roc,thr=',metrics.roc_curve(y_true,y_pred,pos_label=2))
print('fpr=',fpr)
print('tpr=',tpr)
print('thresholds=',thresholds)
print('metrics.auc(fpr,tpr)=', metrics.auc(fpr,tpr))

# y_true=numpy.array([-1,-1,1,1])
# y_pred=numpy.array([0.1,0.4,0.35,0.8])
# fpr,tpr,thresholds=metrics.roc_curve(y_true,y_pred)
# print('metrics.auc(fpr,tpr)=', metrics.auc(fpr,tpr))
# 
# y_true=numpy.array([0,0,1,1])
# y_pred=numpy.array([0.1,0.4,0.35,0.8])
# fpr,tpr,thresholds=metrics.roc_curve(y_true,y_pred)
# print('metrics.auc(fpr,tpr)=', metrics.auc(fpr,tpr))
# 
# y_true=numpy.array([-5,5,10,10])
# y_pred=numpy.array([0.1,0.4,0.35,0.8])
# fpr,tpr,thresholds=metrics.roc_curve(y_true,y_pred,pos_label=10)
# print('metrics.auc(fpr,tpr)=', metrics.auc(fpr,tpr))


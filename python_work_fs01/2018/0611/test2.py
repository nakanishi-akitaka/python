#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:set foldmethod=marker:
# command table 
# zo: open  1 step (o->O, all step)
# zc: close 1 step (c->C, all step)
# zr: open  all fold 1 step (r->R, all step)
# zm: close all fold 1 step (m->M, all step)
# PEP8
################################################################################
# 80 characters / 1 line
################################################################################

# modules
# {{{
import numpy as np
import pandas as pd
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
# }}}

n_test = 60
n_train = 90
raw_data_with_y = pd.read_csv('dataset/iris_withspecies.csv',index_col=0)
raw_data_with_y.iloc[50:, 0] = 'versicolor+virginica'

y = raw_data_with_y.iloc[:, 0]
X = raw_data_with_y.iloc[:, 1:]
if n_train < X.shape[0]:
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=n_test, random_state=0)
else:
    Xtrain = X.copy()
    ytrain = y.copy()
    Xtest = X.copy()
    ytest = y.copy()

autoscaled_Xtrain = (Xtrain - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)
autoscaled_Xtest  = (Xtest  - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)

lda = LinearDiscriminantAnalysis()
lda.fit(autoscaled_Xtrain, ytrain)

calculated_ytrain = lda.predict(autoscaled_Xtrain)
confusion_matrix_train = metrics.confusion_matrix(ytrain, calculated_ytrain, labels=sorted(set(ytrain)))
print('training samples')
print(sorted(set(ytrain)))
print(confusion_matrix_train)

if n_train < X.shape[0]:
    predicted_ytest = lda.predict(autoscaled_Xtest)
    confusion_matrix_test = metrics.confusion_matrix(ytest, predicted_ytest, labels=sorted(set(ytrain)))
    print('')
    print('test samples')
    print(sorted(set(ytrain)))
    print(confusion_matrix_test)

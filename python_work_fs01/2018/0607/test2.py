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
import pandas as pd
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# }}}

raw_data_with_y = pd.read_csv('dataset/iris_withspecies.csv',index_col=0)
raw_data_with_y.iloc[50:,0] = 'versicolor+virginica'
y = raw_data_with_y.iloc[:,0]
X = raw_data_with_y.iloc[:,1:]

scaled_X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

lda = LinearDiscriminantAnalysis()
lda.fit(scaled_X, y)
calculated_y = lda.predict(scaled_X)
confusion_matrix = metrics.confusion_matrix(y, calculated_y, labels=sorted(set(y)))
print(sorted(set(y)))
print(confusion_matrix)

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
# }}}

name = 'dataset/iris.csv'
data = np.array(pd.read_csv(name))[:,:]
y=data[:,0]
X=data[:,1:]
print('covariance')
print(np.cov(X, rowvar=0, bias=0))
print('correlation coefficient')
print(np.corrcoef(X, rowvar=0))

pddata = pd.read_csv(name,index_col=0)
covariance = pddata.cov()
covariance.to_csv('covariance.csv')
correlation_coefficient = pddata.corr()
correlation_coefficient.to_csv('correlation_coefficient.csv')

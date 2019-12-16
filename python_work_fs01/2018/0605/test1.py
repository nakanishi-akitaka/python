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
for i in range(4):
    print(X[:,i].max(), X[:,i].min(), X[:,i].mean(), np.median(X[:,i]), X[:,i].var(ddof=1), X[:,i].std(ddof=1))
pddata = pd.read_csv(name,index_col=0)
basic_statistics = pd.concat([pddata.max() ,pddata.min() ,pddata.mean() ,pddata.median() ,pddata.var(ddof=1) ,pddata.std(ddof=1)],axis=1)
basic_statistics.columns = ['max', 'min', 'average', 'median', 'var', 'std']
basic_statistics.to_csv('basic_statistics.csv')

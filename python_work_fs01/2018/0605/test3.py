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
from sklearn.preprocessing import StandardScaler
# }}}

name = 'dataset/iris.csv'
data = np.array(pd.read_csv(name))[:,:]
y=data[:,0]
X=data[:,1:]
X_scaled=X[:,:]
for i in range(4):
   X_copy = X[:,i]
   X_temp = (X_copy - X_copy.mean()) / X_copy.std(ddof=1)
   X_scaled[:,i] = X_temp[:]
for i in range(len(X_scaled[:,0])):
   print(X_scaled[i,0], X_scaled[i,1], X_scaled[i,2], X_scaled[i,3])

pddata = pd.read_csv(name,index_col=0)
autoscaled_data = (pddata - pddata.mean(axis=0)) / pddata.std(axis=0, ddof=1) 
autoscaled_data.to_csv('autoscaled_data.csv')

print('sklearn')
ss = StandardScaler()
ss.fit(X)
X_scaled = ss.transform(X)
for i in range(len(X_scaled[:,0])):
   print(X_scaled[i,0], X_scaled[i,1], X_scaled[i,2], X_scaled[i,3])

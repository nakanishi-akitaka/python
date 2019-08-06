# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 23:00:01 2019

@author: Akitaka
"""
from scipy.stats.mstats import gmean

X = [[0,1],[2,2]]
print(gmean(X, axis=0))
print(gmean(X, axis=1))


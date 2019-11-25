#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3.2
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
print(rfr.get_params())

# 3.2.1
param_grid = [
{'C': [1, 10, 100, 1000], 'kernel':['leniear']},
{'C': [1, 10, 100, 1000], 'gamma':[0.001, 0.0001], 'kernel':['rbf']},
]

# 3.2.2
import scipy.stats
param_range = {
'C': scipy.stats.expon(scale=100), 
'gamma': scipy.stats.expon(scale=.1),
'kernel':['rbf'], 
'class_weight':['balanced', None]
}

# 3.2.3
# nothing
# 3.2.3.[1-5]
# nothing

# 3.2.4
# nothing
# 3.2.4.[1-3]
# nothing

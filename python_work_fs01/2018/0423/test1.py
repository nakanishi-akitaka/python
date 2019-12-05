#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:set foldmethod=marker:

# command table 
# zo: open  1 step (o->O, all step)
# zc: close 1 step (c->C, all step)
# zr: open  all fold 1 step (r->R, all step)
# zm: close all fold 1 step (m->M, all step)

# modules # {{{
from pymatgen import Composition
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import uniform
from scipy.stats import expon
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# }}}

# input read from csv file # {{{
trainFile = open("tc.csv","r").readlines()

materials = []
tc        = []
pf3= []

for line in trainFile: 
    split = str.split(line, ',')
    material = Composition(split[0])
    pressure = float(split[4])
    tc.append(float(split[8]))
    features = []
    atomicNo = []
    natom = []

    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
        atomicNo.append(float(element.Z))

    features.extend(atomicNo)
    features.extend(natom)
    features.append(pressure)
    pf3.append(features[:])
#   print(features[:],"\n")
# }}}
X = pf3[:]
y = tc[:]

X_train = X[:]
y_train = y[:]

X_test = [ \
[33.0, 1.0, 1.0,8.0, 100.0], \
[33.0, 1.0, 1.0,8.0, 150.0],  \
[33.0, 1.0, 1.0,8.0, 200.0], \
[33.0, 1.0, 1.0,8.0, 250.0],  \
[33.0, 1.0, 1.0,8.0, 300.0], \
[33.0, 1.0, 1.0,8.0, 350.0],  \
[33.0, 1.0, 1.0,8.0, 400.0], \
[33.0, 1.0, 1.0,8.0, 450.0],  \
[33.0, 1.0, 1.0,8.0, 500.0], \
[33.0, 1.0, 1.0,8.0, 550.0],  \
]


def print_score(y_true, y_pred): # {{{
    rmse =  np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    rmae  = np.sqrt(mean_squared_error(y_true, y_pred))/mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("RMSE, MAE, RMSE/MAE, R^2 = %.2f, %.2f, %.2f, %.2f" % (rmse, mae, rmae, r2))
#}}}
#
# 1. Basic follow of machine learning
# {{{
print('# SVR(rbf) with default hyper parameters')
# step 1. model
svr = SVR() 

# step 2. learning
print('learning')
svr.fit(X_train, y_train)
y_temp = svr.predict(X_train)
print_score(y_train,y_temp)

# step 3. predict
y_pred = svr.predict(X_test)

# step 4. score
print('Tc (predicted)',y_pred)
# }}}

#
# 2. parameter optimization (Grid Search)
# {{{
print('')
print('# SVR with GridSearched hyper parameters')
# step 1. model
svr = SVR()

# step 2. learning with optimized parameters
# search range
print('learning')
range_c =  [i*10**j for j in range(-4,5) for i in range(1,2)]
range_g =  [i*10**j for j in range(-4,5) for i in range(1,2)]
# param_grid = [
# {'kernel':['linear'], 'C': range_c},
# {'kernel':['rbf'],    'C': range_c, 'gamma': range_g},
# {'kernel':['sigmoid'],'C': range_c, 'gamma': range_g}]
param_grid = {'C': range_c, 'gamma': range_g}

scores = ['neg_mean_absolute_error','neg_mean_squared_error','r2']
for score in scores:
    print('')
    print('')
    print('scoring = ', score)
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring=score, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    y_temp = grid_search.predict(X_train)
    print_score(y_train,y_temp)
    # step 3. predict
    y_pred = grid_search.predict(X_test)
    # step 4. score
#   means = grid_search.cv_results_['mean_test_score']
#   stds  = grid_search.cv_results_['std_test_score']
#   for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
#       print("%0.3f (+/-%0.03f) for %r" % (mean, std*2, params))
    print('Tc (predicted)',y_pred)
    print("best_score  :", grid_search.best_score_)
    print("best_params :", grid_search.best_params_)
#
# 3. use pipeline, Standard Scaler, PCA 
# 
print('')
print('# SVR with GridSearched hyper parameters after Standardization and PCA')
# step 1. model using pipeline
pipe = Pipeline([
('scaler', StandardScaler()),
('pca', PCA()),
('svr', SVR())
])
# step 2. learning with optimized parameters
# search range
param_grid = {
'svr__C'     : range_c, 
'svr__gamma' : range_g 
}

scores = ['neg_mean_absolute_error','neg_mean_squared_error','r2']
for score in scores:
    print('')
    print('')
    print('scoring = ', score)
    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring=score, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    y_temp = grid_search.predict(X_train)
    print_score(y_train,y_temp)
    # step 3. predict
    y_pred = grid_search.predict(X_test)
    # step 4. score
#   means = grid_search.cv_results_['mean_test_score']
#   stds  = grid_search.cv_results_['std_test_score']
#   for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
#       print("%0.3f (+/-%0.03f) for %r" % (mean, std*2, params))
    print('Tc (predicted)',y_pred)
    print("best_score  :", grid_search.best_score_)
    print("best_params :", grid_search.best_params_)
# }}}

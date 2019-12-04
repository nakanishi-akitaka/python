#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ref
# ../0412/test8.py
# https://qiita.com/ishizakiiii/items/0650723cc2b4eef2c1cf
# ../0413/test5.py
# ../0416/test1.py
# ../0417/test1.py
# ../0413/test3.py
# ../0419/test2.py
#
from sklearn.svm import SVR
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#
# functions printing score
#
def print_prediction_score(y_test,y_pred):
    rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    rmae  = np.sqrt(mean_squared_error(y_test, y_pred))/mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred)
    print('prediction score')
    print('RMSE, MAE, RMSE/MAE, R^2 = %.2f, %.2f, %.2f, %.2f' % (rmse, mae, rmae, r2))

def print_search_score(grid_search):
    print('search score')
    print('best_score  :', grid_search.best_score_)
    print('best_params :', grid_search.best_params_)
#   means = grid_search.cv_results_['mean_test_score']
#   stds  = grid_search.cv_results_['std_test_score']
#   for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
#       print('%0.3f (+/-%0.03f) for %r' % (mean, std*2, params))
#   or 
#   print('grid_scores :', grid_search.grid_scores_)

#
# training data: y = sin(x) + noise
#
X_train = np.sort(5 * np.random.rand(40, 1), axis=0)
y_train = np.sin(X_train).ravel()
y_train[::5] += 3 * (0.5 - np.random.rand(8))
#
# test data: y = sin(x)
#
X_test = X_train[:]
y_test = np.sin(X_test).ravel()
#
# 1. Basic follow of machine learning
#    SVR with default hyper parameters
#
print('')
print('')
print('# 1. SVR with default hyper parameters')

# step 1. model
mod = SVR() 

# step 2. learning
mod.fit(X_train, y_train)

# step 3. predict
y_pred = mod.predict(X_test)

# step 4. score
print_prediction_score(y_test, y_pred)

#
# 2. parameter optimization (Grid Search)
#
print('')
print('')
print('# 2. parameter optimization (Grid Search)')

# step 1. model
mod = SVR() 

# step 2. learning with optimized parameters
# search range
range_c =  [i*10**j for j in range(-2,2) for i in range(1,10)]
range_g =  [i*10**j for j in range(-2,2) for i in range(1,10)]

param_grid = [
{'kernel':['linear'], 'C': range_c},
{'kernel':['rbf'],    'C': range_c, 'gamma': range_g},
{'kernel':['sigmoid'],'C': range_c, 'gamma': range_g}]

grid_search = GridSearchCV(mod, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print_search_score(grid_search)

# step 3. predict
y_pred = grid_search.predict(X_test)

# step 4. score
print_prediction_score(y_test, y_pred)

#
# 3. use pipeline, Standard Scaler, PCA 
# 
print('')
print('')
print('# 3. use pipeline, Standard Scaler, PCA')

# step 1. model using pipeline
pipe = Pipeline([
('scaler', StandardScaler()),
('pca', PCA()),
('svr', SVR())
])

# step 2. learning with optimized parameters
# search range
param_grid = [
{'svr__kernel':['linear'], 'svr__C': range_c},  
{'svr__kernel':['rbf'],    'svr__C': range_c, 'svr__gamma': range_g},
{'svr__kernel':['sigmoid'],'svr__C': range_c, 'svr__gamma': range_g}]
grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print_search_score(grid_search)

# step 3. predict
y_pred = grid_search.predict(X_test)

# step 4. score
print_prediction_score(y_test, y_pred)


#
# 4. many scoring methods
#
print('')
print('')
print('# 4. many scoring methods')

# step 1. model
mod = SVR() 

# search range
param_grid = [
{'kernel':['linear'], 'C': range_c},
{'kernel':['rbf'],    'C': range_c, 'gamma': range_g},
{'kernel':['sigmoid'],'C': range_c, 'gamma': range_g}]


scores = ['neg_mean_absolute_error','neg_mean_squared_error','r2']
for score in scores:
    print('scoring = ', score)
    # step 2. learning with optimized parameters
    grid_search = GridSearchCV(mod, param_grid, cv=5, scoring=score, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print_search_score(grid_search)
    # step 3. predict
    y_pred = grid_search.predict(X_test)
    # step 4. score
    print_prediction_score(y_test, y_pred)
    print('')

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_boston
from pymatgen import Composition, Element
from numpy import zeros, mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model, metrics, ensemble
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
trainFile = open("tc.csv","r").readlines()
#
# function print score of learning and prediction 
# {{{
def print_score(mod, X, y_test):
    y_pred = mod.predict(X)
    rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    rmae  = np.sqrt(mean_squared_error(y_test, y_pred))/mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred)
    print("RMSE, MAE, RMSE/MAE, R^2 = %.3f, %.3f, %.3f, %.3f" % (rmse, mae, rmae, r2))
# }}}
#
# function print score of grid search
# {{{
def print_search_score(grid_search):
    print('search score')
    print('best_score  :', grid_search.best_score_)
    print('best_params :', grid_search.best_params_)
#   means = grid_search.cv_results_['mean_test_score']
#   stds  = grid_search.cv_results_['std_test_score']
#   for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
#       print('%0.3f (+/-%0.03f) for %r' % (mean, std*2, params))
# }}}

tc        = []
pf = []

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
    pf.append(features[:])

X = pf[:]
y = tc[:]
X_train = X[:]
y_train = y[:]

# from sklearn.model_selection import train_test_split
# (X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size = 0.3,random_state = 666)
materials = []
xatom=Element("H")
for i in range(3,86):
    if(not xatom.from_Z(i).is_noble_gas):
        for iatom1 in range(1,10):
            for iatom2 in range(1,10):
#               print('%s%.1i%s%.1i' % (xatom.from_Z(i).symbol,iatom1,xatom.symbol,iatom2))
                str_mat=str(xatom.from_Z(i).symbol)+str(iatom1)+str(xatom.symbol)+str(iatom2)
                materials.append(Composition(str_mat))

X_test = []
for material in materials:
    atomicNo = []
    natom = []
    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
        atomicNo.append(float(element.Z))
#   atom0=element.from_Z(atomicNo[0]).symbol
#   atom1=element.from_Z(atomicNo[1]).symbol
#   print('%s%.1i %s%.1i' % (atom0,natom[0],atom1,natom[1]))
#   print('%s%.1i %s%.1i' % (element.from_Z(atomicNo[0]).symbol,natom[0], \
#                            element.from_Z(atomicNo[1]).symbol,natom[1]))
    for ip in range( 50,550,50):
        temp = []
        temp.extend(atomicNo)
        temp.extend(natom)
        temp.append(float(ip))
        X_test.append(temp[:])
#       print(temp)

#
# 1. parameter optimization (Grid Search)
# {{{
print('')
print('# SVR with GridSearched hyper parameters')
# step 1. model
svr = SVR()

# step 2. learning with optimized parameters
# search range
range_c =  [i*10**j for j in range(-2,2) for i in range(1,10)]
range_g =  [i*10**j for j in range(-2,2) for i in range(1,10)]
range_d =  [i for i in range(1,3)]
param_grid = [
{'kernel':['linear'], 'C': range_c},
{'kernel':['rbf'],    'C': range_c, 'gamma': range_g},
{'kernel':['sigmoid'],'C': range_c, 'gamma': range_g}]
# {'kernel':['poly'],   'C': range_c, 'gamma': range_g, 'degree':range_d}]

grid_search = GridSearchCV(svr,param_grid, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)
print_search_score(grid_search)
print("learning   score: ",end="")
print_score(grid_search, X_train, y_train)

# step 3. predict
y_pred = grid_search.predict(X_test)

for i in range(len(X_test)):
    if(y_pred[i]> 100):
        atom0=element.from_Z(int(X_test[i][0])).symbol
        atom1=element.from_Z(int(X_test[i][1])).symbol
        print('%2s%.1i%1s%.1i P = %.3i GPa Tc = %.1f K' \
        % (atom0,X_test[i][2],atom1,X_test[i][3],int(X_test[i][4]),y_pred[i]))
#
# 3. use pipeline, Standard Scaler, PCA 
# {{{
print('')
print('')
print('# SVR with GridSearched hyper parameters after Standardization and PCA')

# step 1. model using pipeline
pipe = Pipeline([
('scaler', MinMaxScaler()),
('pca', PCA()),
('svr', SVR())
])

# step 2. learning with optimized parameters
# search range
param_grid = [
{'svr__kernel':['linear'], 'svr__C': range_c},  
{'svr__kernel':['rbf'],    'svr__C': range_c, 'svr__gamma': range_g},
{'svr__kernel':['sigmoid'],'svr__C': range_c, 'svr__gamma': range_g}]
grid_search = GridSearchCV(pipe, param_grid, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)
print_search_score(grid_search)
print("learning   score: ",end="")
print_score(grid_search, X_train, y_train)

# step 3. predict
y_pred = grid_search.predict(X_test)

for i in range(len(X_test)):
    if(y_pred[i]> 100):
        atom0=element.from_Z(int(X_test[i][0])).symbol
        atom1=element.from_Z(int(X_test[i][1])).symbol
        print('%2s%.1i%1s%.1i P = %.3i GPa Tc = %.1f K' \
        % (atom0,X_test[i][2],atom1,X_test[i][3],int(X_test[i][4]),y_pred[i]))
#}}}

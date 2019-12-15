#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:set foldmethod=marker:

# modules
# {{{
import pandas as p
import numpy as np
from time import time
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston
from pymatgen import Composition, Element
from numpy import zeros, mean
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneOut
from sklearn import linear_model, metrics, ensemble
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from sklearn.metrics.pairwise import rbf_kernel
# }}}

#
# function print score of learning and prediction 
# {{{
def print_score(mod, X, y_test):
    y_pred = mod.predict(X)
    rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    rmae  = np.sqrt(mean_squared_error(y_test, y_pred))/mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred) # = mod.score(X,y_test)
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
#
# function print high Tc 
# {{{
def print_high_Tc(X_test,y_pred):
    element=Element("H")
    for i in range(len(X_test)):
        if(y_pred[i]> 150):
            atom0=element.from_Z(int(X_test[i][0])).symbol
            atom1=element.from_Z(int(X_test[i][1])).symbol
            print('%2s%.1i%1s%.1i P = %.3i GPa Tc = %.3i K' \
            % (atom0,X_test[i][2],atom1,X_test[i][3],int(X_test[i][4]),y_pred[i]))
#}}}
#
# function Observed-Predicted Plot (yyplot) 
# {{{
def yyplot(y_obs,y_pred):
    yvalues = np.concatenate([y_obs, y_pred])
    ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
    fig = plt.figure(figsize=(8,8))
    plt.scatter(y_obs, y_pred)
    plt.plot([ymin-yrange*0.01, ymax+yrange*0.01],[ymin-yrange*0.01, ymax+yrange*0.01])
    plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.xlabel('y_observed', fontsize=24)
    plt.ylabel('y_predicted', fontsize=24)
    plt.title('Observed-Predicted Plot', fontsize=24)
    plt.tick_params(labelsize=16)
    plt.show()
    return fig
# }}}
#
# read data from csv file
# {{{
def read_file(name): 
    data = np.array(p.read_csv(filepath_or_buffer=name,header=None,sep=','))[:,:]
    y = data[:,0]
    X = data[:,1:]
    return X, y
# }}}
#
# set X_train, y_train, X_test
# {{{
train_file = input('train_file = \n')
test_file  = input('test_file  = \n')
X_train, y_train = read_file(train_file)
X_test,  y_test  = read_file(test_file)
# print(X_train)
# print(y_train)
# print(X_train)
# print(y_test)

# print('print X_train range')
# for i in range(len(X_train[0])):
#     print(X_train[:,i].min(),X_train[:,i].max(),X_train[:,i].mean(),X_train[:,i].std())
# exit()

# }}}
start = time()
# set parameters
# scores = ['neg_mean_absolute_error','neg_mean_squared_error','r2']
score='r2'
range_c = [2**j for j in range(-5,10) ]
range_e = [2**j for j in range(-10,1)]
range_g = [2**j for j in range(-20,10)]
param_grid = [{'svr__kernel':['rbf'], 'svr__C': range_c, 'svr__gamma': range_g, 'svr__epsilon': range_e}]

ss=StandardScaler()
ss.fit(X_train)
ss.transform(X_train)
# Optimize gamma by maximizing variance in Gram matrix
varianceofgrammatrix = list()
for svrgamma in range_g:
    grammatrix = rbf_kernel(X_train, gamma=svrgamma)
    varianceofgrammatrix.append(grammatrix.var(ddof=1))
#   print('gamma, varianceofgrammatrix = ',svrgamma, grammatrix.var(ddof=1)) 
opted_g = range_g[ np.where( varianceofgrammatrix == np.max(varianceofgrammatrix) )[0][0] ]
# print('opted_g      = ', opted_g)


cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
# Optimize epsilon with cross-validation
model = GridSearchCV(SVR(kernel='rbf', C=3, gamma=opted_g), {'epsilon':range_e}, cv=cv )
model.fit(X_train, y_train)
opted_e = model.best_params_['epsilon']

# Optimize C with cross-validation
model = GridSearchCV(SVR(kernel='rbf', epsilon=opted_e, gamma=opted_g), {'C':range_c}, cv=cv )
model.fit(X_train, y_train)
opted_c = model.best_params_['C']

# Optimize gamma with cross-validation (optional)
model = GridSearchCV(SVR(kernel='rbf', epsilon=opted_e, C=opted_c), {'gamma':range_g}, cv=cv )
model.fit(X_train, y_train)
opted_g = model.best_params_['gamma']

# Check optimized hyperparameters
print ("C: {0}, Epsilon: {1}, Gamma: {2}".format(opted_c, opted_e, opted_g))

# Construct SVR model
model = SVR(kernel='rbf', C=opted_c, epsilon=opted_e, gamma=opted_g)
model.fit(X_train, y_train)

# score
print("learning   score: ",end="")
print_score(model, X_train, y_train)
y_pred = model.predict(X_train)
fig=yyplot(y_train,y_pred)

# step 3. predict
# y_pred = grid_search.predict(X_test)
# print_high_Tc(X_test,y_pred)
print("%.2f seconds " % (time() - start))

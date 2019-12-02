#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

trainFile = open("kaisha.csv","r").readlines()
X = []
y = []
X_train = []
X_test  = []
y_train = []
y_test  = []
i = 1
for line in trainFile:
    split = str.split(line, ',')
    rieki = float(split[1])
    sihon = float(split[2])
    ninzu = float(split[3])
    nensu = float(split[4])
    setsumei = [sihon, ninzu, nensu]
    mokuteki = rieki
    all = [rieki, sihon, ninzu, nensu]
    X.append(all)
    if(i%5==3):
        X_test.append(setsumei)
        y_test.append(mokuteki)
    else:
        X_train.append(setsumei)
        y_train.append(mokuteki)
    i+=1
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#
# 1. Basic follow of machine learning
#
# step 1. model
svr = SVR() 

# step 2. learning
svr.fit(X_train, y_train)

# step 3. predict
y_pred = svr.predict(X_test)

# step 4. score
print('# SVR(rbf) with default hyper parameters')
def print_score(y_test,y_pred):
    rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    rmae  = np.sqrt(mean_squared_error(y_test, y_pred))/mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred)
    print("RMSE, MAE, RMSE/MAE, R^2 = %.2f, %.2f, %.2f, %.2f" % (rmse, mae, rmae, r2))
print_score(y_test,y_pred)

#
# 2. parameter optimization (Grid Search)
#
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

grid_search = GridSearchCV(svr,param_grid,cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# step 3. predict
y_pred = grid_search.predict(X_test)

# step 4. score
print('')
print('# SVR with GridSearched hyper parameters')
means = grid_search.cv_results_['mean_test_score']
stds  = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std*2, params))


print_score(y_test,y_pred)
print("best_score  :", grid_search.best_score_)
print("best_params :", grid_search.best_params_)
# print("grid_scores :", grid_search.grid_scores_)
exit()
#
# 3. use pipeline, Standard Scaler, PCA 
# 
# step 1. model using pipeline
pipe = Pipeline([
('scaler', MinMaxScaler()),
('pca', PCA()),
('svr', SVR())
])
# step 2. learning with optimized parameters
# search range
param_grid = {
'pca__n_components' : [1, 2, 3],
'svr__C'     : range_c, 
'svr__gamma' : range_g 
}
grid_search = GridSearchCV(pipe,param_grid, n_jobs=-1)
grid_search.fit(X_train, y_train)

# step 3. predict
y_pred = grid_search.predict(X_test)

# step 4. score
print('')
print('# SVR with GridSearched hyper parameters after Standardization and PCA')
print_score(y_test,y_pred)
print("best_param :", grid_search.best_params_)
#
# 4. parameter optimization (Randomized Search)
# 
# step 1. model using pipeline
pipe = Pipeline([
('scaler', MinMaxScaler()),
('pca', PCA()),
('svr', SVR())
])

# step 2. learning with optimized parameters
# search range
param_dist = {
'pca__n_components' : [1, 2, 3],
'svr__C'     :expon(loc=0,scale=100), 
'svr__gamma' :expon(loc=0,scale=100)
}
n_iter_search = 100
rand_search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1)
rand_search.fit(X_train, y_train)

# step 3. predict
y_pred = rand_search.predict(X_test)

# step 4. score
print('')
print('# SVR with RandomizedSearched hyper parameters after Standardization and PCA')
print_score(y_test,y_pred)
print("best_param :", rand_search.best_params_)
#
# 5. Observed-Predicted Plot (yyplot) 
#
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

# fig = yyplot(y_test,y_pred)

# [ i*10**j for j in range(-2,2) for i in range(1,10)] 
# = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90]

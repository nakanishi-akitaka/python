#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform

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
    X.append(setsumei)
    y.append(mokuteki)
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
svr = SVR(kernel='rbf') 

# step 2. learning
svr.fit(X_train, y_train)

# step 3. predict
y_pred = svr.predict(X_test)

# step 4. score
print('# SVR(rbf) with default hyper parameters')
print("pred values = ",list(y_pred))
print("true values = ",list(y_test))
print(" MSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))
print(" R^2 = ", svr.score(X_test,y_test))


#
# 2. parameter optimization (Grid Search)
#
# step 1. model
svr = SVR(kernel='rbf') 

# step 2. learning with optimized parameters
# search range
param_grid = {
'C':[0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
'gamma':[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
}
grid_search = GridSearchCV(svr,param_grid,cv=5)
grid_search.fit(X_train, y_train)

# step 3. predict
y_pred = grid_search.predict(X_test)

# step 4. score
print('# SVR(rbf) with GridSearched hyper parameters')
print('best_param : {}'.format(grid_search.best_params_))
print("pred values = ",list(y_pred))
print("true values = ",list(y_test))
print(" MSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))
print(" R^2 = ", grid_search.score(X_test,y_test))
#
# 3. use pipeline, Standard Scaler, PCA 
# 
# step 1. model using pipeline
pipe = Pipeline([
('scaler', StandardScaler()),
('pca', PCA()),
('svr', SVR(kernel='rbf'))
])

# step 2. learning with optimized parameters
# search range
param_grid = {
'pca__n_components' : [1, 2, 3],
'svr__C'     : [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
'svr__gamma' : [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
}
grid_search = GridSearchCV(pipe,param_grid,cv=5)
grid_search.fit(X_train, y_train)

# step 3. predict
y_pred = grid_search.predict(X_test)

# step 4. score
print('# SVR(rbf) with GridSearched hyper parameters after Standardization and PCA')
print('best_param : {}'.format(grid_search.best_params_))
print("pred values = ",list(y_pred))
print("true values = ",list(y_test))
print(" MSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))
print(" R^2 = ", grid_search.score(X_test,y_test))
#
# 4. parameter optimization (Randomized Search)
# 
# step 1. model using pipeline
pipe = Pipeline([
('scaler', StandardScaler()),
('pca', PCA()),
('svr', SVR(kernel='rbf'))
])

# step 2. learning with optimized parameters
# search range
param_dist = {
'pca__n_components' : [1, 2, 3],
'svr__C'     :uniform(loc=0.01,scale=10), 
'svr__gamma' :uniform(loc=0.01,scale=10)
}
n_iter_search = 100
rand_search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter_search)
rand_search.fit(X_train, y_train)

# step 3. predict
y_pred = rand_search.predict(X_test)

# step 4. score
print('# SVR(rbf) with RandomizedSearched hyper parameters after Standardization and PCA')
print('best_param : {}'.format(rand_search.best_params_))
print("pred values = ",list(y_pred))
print("true values = ",list(y_test))
print(" MSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))
print(" R^2 = ", rand_search.score(X_test,y_test))

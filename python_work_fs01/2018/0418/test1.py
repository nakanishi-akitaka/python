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
svr = SVR(kernel='rbf') 

# step 2. learning
svr.fit(X_train, y_train)

# step 3. predict
y_pred = svr.predict(X_test)

# step 4. score
print('# SVR(rbf) with default hyper parameters')
rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
rmae  = np.sqrt(mean_squared_error(y_test, y_pred))/mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)
print("RMSE, MAE, RMSE/MAE, R^2 = %.2f, %.2f, %.2f, %.2f" % (rmse, mae, rmae, r2))


#
# 2. parameter optimization (Grid Search)
#
# step 1. model
svr = SVR(kernel='rbf') 

# step 2. learning with optimized parameters
# search range
# 'C'     : [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
# 'gamma' : [0.1,  0.2,  0.3,  0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
param_grid = {
'C'     : [10**i for i in range(-4,4)],
'gamma' : [10**i for i in range(-4,4)]
}
grid_search = GridSearchCV(svr,param_grid,cv=5)
grid_search.fit(X_train, y_train)

# step 3. predict
y_pred = grid_search.predict(X_test)

# step 4. score
print('# SVR(rbf) with GridSearched hyper parameters')
rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
rmae  = np.sqrt(mean_squared_error(y_test, y_pred))/mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)
print("RMSE, MAE, RMSE/MAE, R^2 = %.2f, %.2f, %.2f, %.2f" % (rmse, mae, rmae, r2))
print("best_param : {}".format(grid_search.best_params_))
#
# 3. use pipeline, Standard Scaler, PCA 
# 
# step 1. model using pipeline
pipe = Pipeline([
('scaler', MinMaxScaler()),
('pca', PCA()),
('svr', SVR(kernel='rbf'))
])
# step 2. learning with optimized parameters
# search range
param_grid = {
'pca__n_components' : [1, 2, 3],
'svr__C'     : [10**i for i in range(-4,4)],
'svr__gamma' : [10**i for i in range(-4,4)]
}
# 'svr__C'     : [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
# 'svr__gamma' : [0.1,  0.2,  0.3,  0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
grid_search = GridSearchCV(pipe,param_grid)
grid_search.fit(X_train, y_train)

# step 3. predict
y_pred = grid_search.predict(X_test)

# step 4. score
print('# SVR(rbf) with GridSearched hyper parameters after Standardization and PCA')
rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
rmae  = np.sqrt(mean_squared_error(y_test, y_pred))/mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)
print("RMSE, MAE, RMSE/MAE, R^2 = %.2f, %.2f, %.2f, %.2f" % (rmse, mae, rmae, r2))
print("best_param : {}".format(grid_search.best_params_))
#
# 4. parameter optimization (Randomized Search)
# 
# step 1. model using pipeline
pipe = Pipeline([
('scaler', MinMaxScaler()),
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
n_iter_search = 1
rand_search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter_search)
rand_search.fit(X_train, y_train)

# step 3. predict
y_pred = rand_search.predict(X_test)

# step 4. score
print('# SVR(rbf) with RandomizedSearched hyper parameters after Standardization and PCA')
rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
rmae  = np.sqrt(mean_squared_error(y_test, y_pred))/mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)
print("RMSE, MAE, RMSE/MAE, R^2 = %.2f, %.2f, %.2f, %.2f" % (rmse, mae, rmae, r2))
print("best_param : {}".format(rand_search.best_params_))
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

fig = yyplot(y_test,y_pred)



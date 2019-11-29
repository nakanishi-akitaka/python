#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ref
# ../0412/test8.py
# https://qiita.com/ishizakiiii/items/0650723cc2b4eef2c1cf
#
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
import numpy as np
import pandas as pd


#
# 1. Basic follow of machine learning
#
print('# 1. Basic follow of machine learning')

# class: 0
df_a = pd.DataFrame({'x1':np.random.randn(100),
                     'x2':np.random.randn(100),
                     'y' :0})
# class: 1
df_b = pd.DataFrame({'x1':np.random.randn(100)+5,
                     'x2':np.random.randn(100)+3,
                     'y' :1})
df = df_a.append(df_b)
X_train, X_test, y_train, y_test = \
train_test_split(df[['x1','x2']], df['y'], test_size=0.2)

# step 1. model
clf = RandomForestClassifier(random_state=0)

# step 2. learning
clf.fit(X_train, y_train)

# step 3. predict
y_pred = clf.predict(X_test)

# step 4. score
print("pred values = ",list(y_pred))
print("true values = ",list(y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))


#
# 2. parameter optimization (Grid Search)
#
print('')
print('')
print('# 2. parameter optimization (Grid Search)')

# step 1. model
rfr=RandomForestClassifier(random_state=0)

# step 2. learning with optimized parameters
# search range
param_grid = {
'n_estimators':[2, 5, 10, 100],
'max_depth':[2, 5, 10, 100, 1000]
}
grid_search = GridSearchCV(rfr,param_grid,cv=5)
grid_search.fit(X_train, y_train)
print('test_score : {}'.format(grid_search.score(X_test,y_test)))
print('best_param : {}'.format(grid_search.best_params_))

# step 3. predict
y_pred = grid_search.predict(X_test)

# step 4. score
print("pred values = ",list(y_pred))
print("true values = ",list(y_test))
print("classification_report = ")
print( classification_report(y_test, y_pred))
print("confusion_matrix = ")
print( confusion_matrix(y_test, y_pred))
print(" MSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))

#
# 3. use pipeline, Standard Scaler, PCA 
# 
print('')
print('')
print('# 3. use pipeline')

# step 1. model using pipeline
pipe = Pipeline([
('scaler', StandardScaler()),
('pca', PCA()),
('rf', RandomForestClassifier(random_state=0))
])

# step 2. learning with optimized parameters
# search range
param_grid = {
'pca__n_components' : [1, 2],
'rf__n_estimators' : [2, 5, 10, 100],
'rf__max_depth' : [2, 5, 10, 100, 1000]
}
grid_search = GridSearchCV(pipe,param_grid,cv=5)
grid_search.fit(X_train, y_train)
print('test_score : {}'.format(grid_search.score(X_test,y_test)))
print('best_param : {}'.format(grid_search.best_params_))

# step 3. predict
y_pred = grid_search.predict(X_test)

# step 4. score
print("pred values = ",list(y_pred))
print("true values = ",list(y_test))
print("classification_report = ")
print( classification_report(y_test, y_pred))
print("confusion_matrix = ")
print( confusion_matrix(y_test, y_pred))
print(" MSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))

#
# 4. parameter optimization (Randomized Search)
# 
print('')
print('')
print('# 4. parameter optimization (Randomized Search)')

# step 1. model using pipeline
pipe = Pipeline([
('scaler', StandardScaler()),
('pca', PCA()),
('rf', RandomForestClassifier(random_state=0))
])

# step 2. learning with optimized parameters
# search range
param_dist = {
'pca__n_components' : [1, 2],
'rf__n_estimators' : sp_randint(2,100),
'rf__max_depth' : sp_randint(2,1000)
}
n_iter_search = 20
rand_search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter_search)
rand_search.fit(X_train, y_train)
print('test_score : {}'.format(rand_search.score(X_test,y_test)))
print('best_param : {}'.format(rand_search.best_params_))

# step 3. predict
y_pred = rand_search.predict(X_test)

# step 4. score
print("pred values = ",list(y_pred))
print("true values = ",list(y_test))
print("classification_report = ")
print( classification_report(y_test, y_pred))
print("confusion_matrix = ")
print( confusion_matrix(y_test, y_pred))
print(" MSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))

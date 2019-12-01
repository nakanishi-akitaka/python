#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import uniform
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
lr = LinearRegression()

# step 2. learning
lr.fit(X_train, y_train)

# step 3. predict
y_pred = lr.predict(X_test)

# step 4. score
print('# LR(rbf) with default hyper parameters')
rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
rmae  = np.sqrt(mean_squared_error(y_test, y_pred))/mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)
print("RMSE, MAE, RMSE/MAE, R^2 = %.2f, %.2f, %.2f, %.2f" % (rmse, mae, rmae, r2))

#
# 2. use pipeline, Standard Scaler, PCA 
# 
# step 1. model using pipeline
pipe = Pipeline([
('scaler', MinMaxScaler()),
('pca', PCA()),
('lr', LinearRegression())
])
# step 2. learning
pipe.fit(X_train, y_train)

# step 3. predict
y_pred = pipe.predict(X_test)

# step 4. score
print('# LR(rbf) after Standardization and PCA')
rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
rmae  = np.sqrt(mean_squared_error(y_test, y_pred))/mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)
print("RMSE, MAE, RMSE/MAE, R^2 = %.2f, %.2f, %.2f, %.2f" % (rmse, mae, rmae, r2))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy  import array
from pandas import DataFrame
from sklearn.datasets import load_boston

boston = load_boston()
df = DataFrame(boston.data,columns = boston.feature_names)
df['MEDV'] = array(boston.target)

X = df.iloc[:, :-1].values
y = df.loc[:, 'MEDV'].values

from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size = 0.3,random_state = 666)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X_train,y_train)

from sklearn.model_selection import GridSearchCV
params = {'n_estimators' : [3, 10, 100, 1000, 10000], 'n_jobs': [-1]}

mod = RandomForestRegressor()
cv = GridSearchCV(mod, params, cv = 10, scoring = 'neg_mean_squared_error', n_jobs = 1)
cv.fit(X_train, y_train)

y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

from sklearn.metrics import mean_squared_error
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
from sklearn.metrics import r2_score
print('R^2 train : %.3f, test : %.3f' % (r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))

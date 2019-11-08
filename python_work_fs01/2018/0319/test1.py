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

y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

from sklearn.metrics import mean_squared_error
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
from sklearn.metrics import r2_score
print('R^2 train : %.3f, test : %.3f' % (r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))

import matplotlib.pyplot as plt
plt.figure(figsize = (10, 7))
plt.scatter(y_train_pred, y_train_pred - y_train, c = 'black', marker = 'o', s=35, alpha = 0.5, label = 'Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c = 'lightgreen', marker = 's', s=35, alpha = 0.7, label = 'Test data')
plt.xlabel('Predicted valued')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'red')
plt.xlim([-10, 50])
# plt.show()

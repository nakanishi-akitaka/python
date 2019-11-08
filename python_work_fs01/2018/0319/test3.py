#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy  import array
from pandas import DataFrame
from sklearn.datasets import load_boston

# data WO import
boston = load_boston()
# SETSUMEI HENNSUU WO DataFrame HE HENKAN
df = DataFrame(boston.data,columns = boston.feature_names)
# MOKUTEKI HENNSUU WO DataFrame He TUIKA
df['MEDV'] = array(boston.target)
# print(df.head())

# SETSUMEI HENNSUU
X = df.loc[:,boston.feature_names].values
# MOKUTEKI HENNSUU
y = df.loc[:, 'MEDV'].values

from sklearn.linear_model import LinearRegression
mod = LinearRegression(fit_intercept = True, normalize = False, copy_X = True, n_jobs = 1)
mod.fit(X,y)

# print(mod.coef_)
# print(mod.intercept_)

from sklearn.model_selection import train_test_split
# 70% = GAKUSYUU-YOU, 30% = KENSYOU-YOU data NI SURUYOU BUNKATSU
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 666)

# GAKUSYUU-YOU data DE parameter SUITEI
mod.fit(X_train, y_train)

# SAKUSEI SHITA model KARA YOSOKU (GAKUSYUU-YOU, KENSYOU-YOU model SHIYOU)
y_train_pred = mod.predict(X_train)
y_test_pred = mod.predict(X_test)

from sklearn.metrics import mean_squared_error
# GAKUSYUU-YOU, KENSYOU-YOU data NI KANSITE HEIKIN-JIJOU-GOSA WO SHUTSURYOKU
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
# GAKUSYUU-YOU, KENSYOU-YOU data NI KANSITE R^2 WO SHUTSURYOKU
print('R^2 train : %.3f, test : %.3f' % (mod.score(X_train,y_train), mod.score(X_test,y_test)))
from sklearn.metrics import r2_score
print('R^2 train : %.3f, test : %.3f' % (r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))

import matplotlib.pyplot as plt
# X-axis = YOSOKU, Y-axis = ZANSA
plt.scatter(y_train_pred, y_train_pred - y_train, c = 'black', marker = 'o', label = 'Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c = 'lightgreen', marker = 's', label = 'Test data')
plt.xlabel('Predicted valued')
plt.ylabel('Residuals')
# HANREI WO HIDARI-UE NI HYOUJI
plt.legend(loc = 'upper left')
# y = 0 NI CHOKUSEN NI HIKU
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'red')
plt.xlim([10, 50])
# plt.show()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy  import array, arange, newaxis, log, sqrt
from pandas import DataFrame
from sklearn.datasets import load_boston

boston = load_boston()
df = DataFrame(boston.data,columns = boston.feature_names)
df['MEDV'] = array(boston.target)

# import matplotlib.pyplot as plt
# import seaborn as sns
# # graph NO style WO SHITEI
# sns.set(style = 'whitegrid', context = 'notebook')
# # HENSUU NO pair NO KANKEI WO plot
# sns.pairplot(df, size = 2.5)
# plt.show()
# # column WO SEIGEN SHITE plot
# sns.pairplot(df[['LSTAT','MEDV']], size = 2.5)
# plt.show()

# SETSUMEI HENNSUU
X = df.loc[:, ['LSTAT']].values
# MOKUTEKI HENNSUU
y = df.loc[:, 'MEDV'].values

from sklearn.linear_model import LinearRegression
mod = LinearRegression()

from sklearn.preprocessing import PolynomialFeatures
# 2 JI HENSUU WO SAKUSEI SURU instance
quadratic = PolynomialFeatures(degree = 2)
# 3 JI HENSUU WO SAKUSEI SURU instance
cubic = PolynomialFeatures(degree = 3)
# HENSUU SAKUSEI
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# model SHIKI-YOU NI HENSUU WO SAKUSEI
X_fit = arange(X.min(), X.max(), 1)[:, newaxis]

# SENKEI-KAIKI model, YOSOKU-CHI, R^2 WO HYOUKA
mod_lin = mod.fit(X, y)
y_lin_fit = mod_lin.predict(X_fit)
r2_lin = mod.score(X, y)

# 2 JI NO KOU WO TSUIKA, YOSOKU-CHI, R^2 WO HYOUKA
mod_quad = mod.fit(X_quad, y)
y_quad_fit = mod_quad.predict(quadratic.fit_transform(X_fit))
r2_quad = mod.score(X_quad, y)

# 3 JI NO KOU WO TSUIKA, YOSOKU-CHI, R^2 WO HYOUKA
mod_cubic = mod.fit(X_cubic, y)
y_cubic_fit = mod_cubic.predict(cubic.fit_transform(X_fit))
r2_cubic = mod.score(X_cubic, y)

# import matplotlib.pyplot as plt
# plt.scatter(X, y, label = 'Training points', color = 'lightgray')
# # 1-JI
# plt.scatter(X_fit, y_lin_fit, \
#  label = 'linear (d = 1), $R^2=%.2f$' % r2_lin, \
#  color = 'blue', lw = 2, linestyle = ':')
# # 2-JI
# plt.scatter(X_fit, y_quad_fit, \
#  label = 'quadratic (d = 2), $R^2=%.2f$' % r2_quad, \
#  color = 'red', lw = 2, linestyle = '-')
# # 3-JI
# plt.scatter(X_fit, y_cubic_fit, \
#  label = 'cubic (d = 3), $R^2=%.2f$' % r2_cubic, \
#  color = 'green', lw = 2, linestyle = '--')
# plt.xlabel('LSTAT')
# plt.ylabel('MEDV')
# plt.legend(loc = 'upper right')
# plt.show()

from sklearn.model_selection import train_test_split
# 70% = GAKUSYUU, 30% = KENSYOU NI BUNKATU
(X_train, X_test, y_train, y_test) = \
train_test_split(X, y , test_size = 0.3, random_state = 666)
(X_quad_train, X_quad_test, y_train, y_test) = \
train_test_split(X_quad, y , test_size = 0.3, random_state = 666)
(X_cubic_train, X_cubic_test, y_train, y_test) = \
train_test_split(X_cubic, y , test_size = 0.3, random_state = 666)

# SENKEI model
mod.fit(X_train, y_train)
y_train_pred = mod.predict(X_train)
y_test_pred = mod.predict(X_test)
from sklearn.metrics import mean_squared_error
# # GAKUSYUU-YOU, KENSYOU-YOU data NI KANSITE HEIKIN-JIJOU-GOSA WO SHUTSURYOKU
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
# # GAKUSYUU-YOU, KENSYOU-YOU data NI KANSITE R^2 WO SHUTSURYOKU
print('R^2 train : %.3f, test : %.3f' % (mod.score(X_train,y_train), mod.score(X_test,y_test)))

# 2-JI NO KOU WO TSUIKA
# mod.fit(X_train, y_train)
mod.fit(X_quad_train, y_train)
y_train_pred = mod.predict(X_quad_train)
y_test_pred = mod.predict(X_quad_test)
from sklearn.metrics import mean_squared_error
# # GAKUSYUU-YOU, KENSYOU-YOU data NI KANSITE HEIKIN-JIJOU-GOSA WO SHUTSURYOKU
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
# # GAKUSYUU-YOU, KENSYOU-YOU data NI KANSITE R^2 WO SHUTSURYOKU
print('R^2 train : %.3f, test : %.3f' % (mod.score(X_quad_train,y_train), mod.score(X_quad_test,y_test)))

# 3-JI NO KOU WO TSUIKA
# mod.fit(X_train, y_train)
mod.fit(X_cubic_train, y_train)
y_train_pred = mod.predict(X_cubic_train)
y_test_pred = mod.predict(X_cubic_test)
from sklearn.metrics import mean_squared_error
# # GAKUSYUU-YOU, KENSYOU-YOU data NI KANSITE HEIKIN-JIJOU-GOSA WO SHUTSURYOKU
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
# # GAKUSYUU-YOU, KENSYOU-YOU data NI KANSITE R^2 WO SHUTSURYOKU
print('R^2 train : %.3f, test : %.3f' % (mod.score(X_cubic_train,y_train), mod.score(X_cubic_test,y_test)))

# HI-SENKEI-HENKAN
X_log = log(X)
y_sqrt = sqrt(y)

# model SAKUSEI
X_fit = arange(X_log.min(), X_log.max() + 1, 1)[:, newaxis]
mod_log = mod.fit(X_log, y_sqrt)
y_sqrt_fit = mod_log.predict(X_fit)
r2_sqrt = mod.score(X_log, y_sqrt)

import matplotlib.pyplot as plt
plt.scatter(X_log, y_sqrt, label = 'Training points', color = 'lightgray')
plt.plot(X_fit, y_sqrt_fit, \
 label = 'linear (d = 1), $R^2=%.2f$' % r2_sqrt, \
 color = 'blue', lw = 2, linestyle = ':')
plt.xlabel('log[LSTAT]')
plt.ylabel('sqrt[MEDV]')
plt.legend(loc = 'upper right')
# plt.show()

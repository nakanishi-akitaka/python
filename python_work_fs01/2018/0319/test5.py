#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy  import array, arange, newaxis, log, sqrt
from pandas import DataFrame
from sklearn.datasets import load_boston

boston = load_boston()
df = DataFrame(boston.data,columns = boston.feature_names)
df['MEDV'] = array(boston.target)


from sklearn.tree import DecisionTreeRegressor
X = df.loc[:, ['LSTAT']].values
y = df.loc[:, 'MEDV'].values
# model KOUCHIKU, KI NOFUKASA HA 3 NI KOTEI
tree = DecisionTreeRegressor(max_depth = 3)
tree.fit(X, y)

# KAIKI-CHOKUSEN WO ZUSHI SURUNONI HENSUU WO NARABIKAE
# sort_idx = X.flatten().argsort()
# import matplotlib.pyplot as plt
# plt.figure(figsize = (10, 7))
# plt.scatter(X[sort_idx], y[sort_idx], c = 'blue', label = 'Training points')
# plt.plot(X[sort_idx], tree.predict(X[sort_idx]), color = 'red', label = 'depth = 3')
# plt.xlabel('LSTAT')
# plt.ylabel('MEDV')
# plt.legend(loc = 'upper right')
# plt.show()

# ARAKAJIME IRO NO list WO SAKUSEI
sort_idx = X.flatten().argsort()
color = ['red', 'green', 'yellow', 'magenta', 'cyan']
import matplotlib.pyplot as plt
plt.figure(figsize = (10, 7))
plt.scatter(X[sort_idx], y[sort_idx], c = 'lightgray', label = 'Training points')
for t in (arange(5)):
    tree = DecisionTreeRegressor(max_depth = t + 1)
    tree.fit(X, y)
    sort_idx = X.flatten().argsort()
    plt.plot(X[sort_idx], tree.predict(X[sort_idx]), color = color[t], label = 'depth = %.1f' %(t + 1))
    plt.xlabel('LSTAT')
    plt.ylabel('MEDV')
    plt.legend(loc = 'upper right')
plt.show()

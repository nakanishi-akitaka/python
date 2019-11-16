#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import datasets
boston = datasets.load_boston()
# print(boston['feature_names'])
# print(boston['data'])
# print(boston['target'])

from sklearn.model_selection import train_test_split
X = boston['data']
y = boston['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor()
regr.fit(X_train, y_train)

y_predicted = regr.predict(X_test)

from sklearn import metrics
R2test = metrics.r2_score(y_test, y_predicted)
print(R2test)

regr = RandomForestRegressor(oob_score=True)
regr.fit(X_train, y_train)

oob_score = regr.oob_score_
print(oob_score)

feature_importances = regr.feature_importances_
print(feature_importances)

import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.ylim([0,1])
y = feature_importances
x = np.arange(len(y))
plt.bar(x,y,align="center")
plt.xticks(x,boston['feature_names'])
plt.show()

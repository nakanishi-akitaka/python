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

# print(metrics.accuracy_score(y_test, y_predicted))
# print(metrics.average_precision_score(y_test, y_predicted))
# print(metrics.precision_score(y_test, y_predicted))
# print(metrics.recall_score(y_test, y_predicted))
# print(metrics.roc_auc_score(y_test, y_predicted))
print(metrics.explained_variance_score(y_test, y_predicted))
print(metrics.mean_absolute_error(y_test, y_predicted))
print(metrics.mean_squared_error(y_test, y_predicted))
print(metrics.mean_squared_log_error(y_test, y_predicted))
print(metrics.r2_score(y_test, y_predicted))

regr = RandomForestRegressor(oob_score=True)
regr.fit(X_train, y_train)
print('regr.estimators_')
print(regr.estimators_)
print('')
print('regr.feature_importances_')
print(regr.feature_importances_)
print('')
print('regr.n_features_')
print(regr.n_features_)
print('')
print('regr.n_outputs_')
print(regr.n_outputs_)
print('')
print('regr.oob_score_')
print(regr.oob_score_)
print('')
print('regr.oob_prediction_')
print(regr.oob_prediction_)



# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,5))
# plt.ylim([0,1])
# y = feature_importances
# x = np.arange(len(y))
# plt.bar(x,y,align="center")
# plt.xticks(x,boston['feature_names'])
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,5))
# plt.ylim([0,1])
# y = feature_importances
# x = np.arange(len(y))
# plt.bar(x,y,align="center")
# plt.xticks(x,boston['feature_names'])
# plt.show()

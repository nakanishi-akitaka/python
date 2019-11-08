#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pymatgen import Composition, Element
from numpy import zeros, mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model, metrics, ensemble
trainFile = open("tc.csv","r").readlines()

materials = []
tc        = []
pf3= []

for line in trainFile:
    split = str.split(line, ',')
    material = Composition(split[0])
    pressure = float(split[4])
    tc.append(float(split[8]))
    features = []
    atomicNo = []
    natom = []

    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
        atomicNo.append(float(element.Z))

    features.extend(atomicNo)
    features.extend(natom)
    features.append(pressure)
    pf3.append(features[:])

# sklearn NO random forest KAIKI
rfr = ensemble.RandomForestRegressor(n_estimators=10)

# KOUSA KENSHO SIMASU
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
sp = cross_val_score(rfr,pf3,tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 3 data: "+ str(round(abs(mean(sp)), 3)) + " K")


X = pf3[:]
y = tc[:]
# print(X)
# print(y)
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
# from sklearn.metrics import r2_score
# print('R^2 train : %.3f, test : %.3f' % (r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))

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
plt.show()

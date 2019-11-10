#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import sqrt
from numpy  import array
from sklearn.datasets import load_boston
from pymatgen import Composition, Element
from numpy import zeros, mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model, metrics, ensemble
trainFile = open("test8.csv","r").readlines()

materials = []
tc        = []
pf        = []

for line in trainFile:
    split = str.split(line, ',')
    material = Composition(split[0])
    spacegroup = float(split[1])
    mass     = 1/sqrt(float(split[2]))
    hrate    = float(split[3])
    pressure = float(split[4])
    lambda0  = float(split[5])
    omegalog = float(split[6])
    tc.append(float(split[8]))
    features = []
    atomicNo = []
    natom = []
    group = []
    row = []

    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
        atomicNo.append(float(element.Z))
        group.append(float(element.group))
        row.append(float(element.row))

    features.append(lambda0)
    features.append(omegalog)
    features.append(spacegroup)
    features.append(pressure)
    features.append(hrate)
    features.append(mass)
    features.extend(group)
    features.extend(row)
    features.extend(natom)
    features.extend(atomicNo)
    pf.append(features[:])
#   print(features[:],"\n")

X = pf[:]
y = tc[:]

from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size = 0.3,random_state = 666)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X_train,y_train)

from sklearn.model_selection import GridSearchCV
# params = {'n_estimators' : [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'n_jobs': [-1]}
params = {'n_estimators' : [5, 10, 15, 20], 'n_jobs': [-1]}

mod = RandomForestRegressor()
cv = GridSearchCV(mod, params, cv = 10, scoring = 'r2', n_jobs = 1)
cv.fit(X_train, y_train)

y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

# from sklearn.metrics import mean_squared_error
# print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
from sklearn.metrics import r2_score
print('R^2 train : %.3f, test : %.3f' % (r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))

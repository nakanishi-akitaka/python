#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy  import array
from sklearn.datasets import load_boston
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
#   print(features[:],"\n")

X = pf3[:]
y = tc[:]

from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size = 0.3,random_state = 666)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print('n_estimators')
for ns in range(2,21,1):
    forest = RandomForestRegressor(n_estimators=ns)
    forest.fit(X_train,y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    print('%.2i, %.3f, %.3f' % (ns, r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))
#   print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
#   print('R^2 train : %.3f, test : %.3f' % (r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))
print('max_depth')
for md in range(2,21,1):
    forest = RandomForestRegressor(max_depth=md)
    forest.fit(X_train,y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    print('%.2i, %.3f, %.3f' % (md, r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))
#   print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
#   print('R^2 train : %.3f, test : %.3f' % (r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))

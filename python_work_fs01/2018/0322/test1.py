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

# sklearn NO random forest KAIKI
rfr = ensemble.RandomForestRegressor(n_estimators=10)

# KOUSA KENSHO SIMASU
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
sp = cross_val_score(rfr,pf3,tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 3 data: "+ str(round(abs(mean(sp)), 3)) + " K")


X = pf3[:]
y = tc[:]

# from sklearn.model_selection import train_test_split
# (X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size = 0.3,random_state = 666)
X_train = X[:]
y_train = y[:]

X_test = [ \
[33.0, 1.0, 1.0,8.0, 100.0], \
[33.0, 1.0, 1.0,8.0, 150.0],  \
[33.0, 1.0, 1.0,8.0, 200.0], \
[33.0, 1.0, 1.0,8.0, 250.0],  \
[33.0, 1.0, 1.0,8.0, 300.0], \
[33.0, 1.0, 1.0,8.0, 350.0],  \
[33.0, 1.0, 1.0,8.0, 400.0], \
[33.0, 1.0, 1.0,8.0, 450.0],  \
[33.0, 1.0, 1.0,8.0, 500.0], \
[33.0, 1.0, 1.0,8.0, 550.0],  \
]

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X_train,y_train)

y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
print("y_test_pred",y_test_pred)

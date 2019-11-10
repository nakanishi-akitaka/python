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

tc        = []
pf = []

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
    pf.append(features[:])

X = pf[:]
y = tc[:]

# from sklearn.model_selection import train_test_split
# (X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size = 0.3,random_state = 666)
X_train = X[:]
y_train = y[:]
materials = []
for str_mat in ["AsH8","SH3"]:
    materials.append(Composition(str_mat))
X_test = []
for material in materials:
    atomicNo = []
    natom = []
    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
        atomicNo.append(float(element.Z))
#   atom0=element.from_Z(atomicNo[0]).symbol
#   atom1=element.from_Z(atomicNo[1]).symbol
#   print('%s%.1i %s%.1i' % (atom0,natom[0],atom1,natom[1]))
#   print('%s%.1i %s%.1i' % (element.from_Z(atomicNo[0]).symbol,natom[0], \
#                            element.from_Z(atomicNo[1]).symbol,natom[1]))
    for ip in range(100,300,50):
        temp = []
        temp.extend(atomicNo)
        temp.extend(natom)
        temp.append(float(ip))
        X_test.append(temp[:])
#       print(temp)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X_train,y_train)

y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
for i in range(len(X_test)):
    atom0=element.from_Z(int(X_test[i][0])).symbol
    atom1=element.from_Z(int(X_test[i][1])).symbol
    print('%2s%.1i%1s%.1i P = %.3i GPa Tc = %.1f K' \
    % (atom0,X_test[i][2],atom1,X_test[i][3],int(X_test[i][4]),y_test_pred[i]))
#   print(X_test[i], y_test_pred[i])

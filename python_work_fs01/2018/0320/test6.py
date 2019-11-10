#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy  import array
from sklearn.datasets import load_boston
from pymatgen import Composition, Element
from numpy import zeros, mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model, metrics, ensemble
trainFile = open("tc_noAr.csv","r").readlines()

materials = []
tc        = []
pf= [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

for line in trainFile:
    split = str.split(line, ',')
    material = Composition(split[0])
    pressure = float(split[4])
    tc.append(float(split[8]))
    features = []
    feature = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

    for element in material:
        feature[ 1].append(float(element.Z))
        feature[ 2].append(material.get_atomic_fraction(element)*material.num_atoms)
        feature[ 3].append(float(element.group))
        feature[ 4].append(float(element.row))
        feature[ 5].append(element.X)
        feature[ 6].append(float(element.max_oxidation_state))
        feature[ 7].append(float(element.min_oxidation_state))
        feature[ 8].append(float(str(element.atomic_mass).split("a")[0]))
        feature[ 9].append(float(element.mendeleev_no))
        feature[10].append(float(str(element.melting_point).split("K")[0]))
        feature[11].append(float(str(element.molar_volume).split("c")[0]))
        feature[12].append(float(str(element.thermal_conductivity).split("W")[0]))
        feature[13].append(element.is_noble_gas)
        feature[14].append(element.is_transition_metal)
        feature[15].append(element.is_rare_earth_metal)
        feature[16].append(element.is_metalloid)
        feature[17].append(element.is_alkali)
        feature[18].append(element.is_alkaline)
        feature[19].append(element.is_halogen)
        feature[20].append(element.is_chalcogen)
        feature[21].append(element.is_lanthanoid)
        feature[22].append(element.is_actinoid)
        feature[23].append(pressure)

    for i in range(1,24):
        features.extend(feature[i])
        pf[i].append(features[:])

# sklearn NO random forest KAIKI
rfr = ensemble.RandomForestRegressor(n_estimators=10)

# # KOUSA KENSHO SIMASU
# cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
# sp = cross_val_score(rfr,pf[23],tc, cv=cv, scoring='neg_mean_absolute_error')
# print("MAE by RF with physical 23 data: ave ", round(sc.mean(), 3)," std ", round(sc.std(), 3))
# sp = cross_val_score(rfr,pf[23],tc, cv=cv, scoring='r2')
# print("R2  by RF with physical 23 data: ave ", round(sc.mean(), 3)," std ", round(sc.std(), 3))

X_train = pf[23]
y_train = tc[:]

trainFile = open("test5.csv","r").readlines()

materials = []
tc        = []
pf= [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

for line in trainFile:
    split = str.split(line, ',')
    material = Composition(split[0])
    pressure = float(split[4])
    tc.append(float(split[8]))
    features = []
    feature = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

    for element in material:
        feature[ 1].append(float(element.Z))
        feature[ 2].append(material.get_atomic_fraction(element)*material.num_atoms)
        feature[ 3].append(float(element.group))
        feature[ 4].append(float(element.row))
        feature[ 5].append(element.X)
        feature[ 6].append(float(element.max_oxidation_state))
        feature[ 7].append(float(element.min_oxidation_state))
        feature[ 8].append(float(str(element.atomic_mass).split("a")[0]))
        feature[ 9].append(float(element.mendeleev_no))
        feature[10].append(float(str(element.melting_point).split("K")[0]))
        feature[11].append(float(str(element.molar_volume).split("c")[0]))
        feature[12].append(float(str(element.thermal_conductivity).split("W")[0]))
        feature[13].append(element.is_noble_gas)
        feature[14].append(element.is_transition_metal)
        feature[15].append(element.is_rare_earth_metal)
        feature[16].append(element.is_metalloid)
        feature[17].append(element.is_alkali)
        feature[18].append(element.is_alkaline)
        feature[19].append(element.is_halogen)
        feature[20].append(element.is_chalcogen)
        feature[21].append(element.is_lanthanoid)
        feature[22].append(element.is_actinoid)
        feature[23].append(pressure)

    for i in range(1,24):
        features.extend(feature[i])
        pf[i].append(features[:])

X_test = pf[23]
y_test = tc[:]

# from sklearn.ensemble import RandomForestRegressor
# forest = RandomForestRegressor()
# forest.fit(X_train,y_train)
from sklearn.linear_model import LinearRegression
mod = LinearRegression(fit_intercept = True, normalize = False, copy_X = True, n_jobs = 1)
mod.fit(X_train,y_train)
# print("X_train",X_train)
# print("y_train",y_train)

# from sklearn.model_selection import GridSearchCV
# params = {'n_estimators' : [3, 10, 100, 1000, 10000], 'n_jobs': [-1]}
# 
# mod = RandomForestRegressor()
# cv = GridSearchCV(mod, params, cv = 10, scoring = 'neg_mean_squared_error', n_jobs = 1)
# cv.fit(X_train, y_train)

y_train_pred = mod.predict(X_train)
y_test_pred = mod.predict(X_test)
# print("X_test",X_test)
print("y_test",y_test)
print("y_test_pred",y_test_pred)

from sklearn.metrics import mean_squared_error
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
from sklearn.metrics import r2_score
print('R^2 train : %.3f, test : %.3f' % (r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pymatgen import Composition, Element
from numpy import zeros, mean
trainFile = open("tc.csv","r").readlines()

materials = []
tc        = []

for line in trainFile:
    split = str.split(line, ',')
    material = Composition(split[0])
    materials.append(material)
    tc.append(float(split[8]))

error = mean(abs(mean(tc) - tc))
print("MAE:  " + str(round(error, 3)) + " K")



from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model, metrics, ensemble

# sklearn NO random forest KAIKI
rfr = ensemble.RandomForestRegressor(n_estimators=10)

# KOUSA KENSHO SIMASU
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)


pf1= []
pf2= []
pf3= []
pf4= []
pf5= []
pf6= []
pf7= []
pf8= []
pf9= []
pf10= []
pf11= []
pf12= []
pf13= []
pf14= []
pf15= []
pf16= []
pf17= []
pf18= []
pf19= []
pf20= []
pf21= []
pf22= []
pf23= []
pf24= []

for line in trainFile:
    split = str.split(line, ',')
    material = Composition(split[0])
    spacegroup = float(split[1])
    hrate    = float(split[3])
    pressure = float(split[4])
    features = []
    atomicNo = []
    natom = []
    eneg = []
    group = []
    row = []
    amass = []

    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
        atomicNo.append(float(element.Z))
        group.append(float(element.group))
        row.append(float(element.row))
        amass.append(float(str(element.atomic_mass).split("a")[0]))

    pf1.append(atomicNo[:])
    pf2.append(natom[:])
    pf3.append(group[:])
    pf4.append(row[:])
    pf5.append([spacegroup,pressure])
    pf6.append(amass[:])

sp = cross_val_score(rfr,pf1,tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 1 data: "+ str(round(abs(mean(sp)), 3)) + " K")
sp = cross_val_score(rfr,pf2,tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 2 data: "+ str(round(abs(mean(sp)), 3)) + " K")
sp = cross_val_score(rfr,pf3,tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 3 data: "+ str(round(abs(mean(sp)), 3)) + " K")
sp = cross_val_score(rfr,pf4,tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 4 data: "+ str(round(abs(mean(sp)), 3)) + " K")
sp = cross_val_score(rfr,pf5,tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 5 data: "+ str(round(abs(mean(sp)), 3)) + " K")
sp = cross_val_score(rfr,pf6,tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 6 data: "+ str(round(abs(mean(sp)), 3)) + " K")

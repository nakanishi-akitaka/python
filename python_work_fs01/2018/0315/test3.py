#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from pymatgen import Composition, Element
from numpy import zeros, mean
trainFile = open("lambda.csv","r").readlines()
# trainFile = open("test1.csv","r").readlines()

# input: pymatgen NO Composition object
# output: SOSEI vector
def naiveVectorize(composition):
    vector = zeros((MAX_Z))
    for element in composition:
        # element HA GENSI. fraction HA SONO GENSI GA SOSEI NI HUKUMARERU WARIAI
        fraction = composition.get_atomic_fraction(element)
        vector[element.Z - 1] = fraction
    return(vector)

materials = []
tc = []
naiveFeatures = []

MAX_Z = 100

for line in trainFile:
    split = str.split(line, ',')
    material = Composition(split[0])
    materials.append(material)
    naiveFeatures.append(naiveVectorize(material))
#   print(material,float(split[8]))
    tc.append(float(split[5]))

error = mean(abs(mean(tc) - tc))
print("MAE:  " + str(round(error, 3)) + "  ")




from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model, metrics, ensemble

# sklearn NO random forest KAIKI
rfr = ensemble.RandomForestRegressor(n_estimators=10)

# KOUSA KENSHO SIMASU
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

sc = cross_val_score(rfr, naiveFeatures, tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with composition data: "+ str(round(abs(mean(sc)), 3)) + "  ")


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

for line in trainFile:
    split = str.split(line, ',')
    material = Composition(split[0])
    spacegroup = float(split[1])
    pressure = float(split[4])
    features = []
    fraction = []
    atomicNo = []
    natom = []
    eneg = []
    group = []
    row = []
    maxos = []
    minos = []
    amass = []
    aradi = []
    mende = []

    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
####    print(material,natom,element)
        atomicNo.append(float(element.Z))
        group.append(float(element.group))
        row.append(float(element.row))
#       eneg.append(element.X)
        maxos.append(float(element.max_oxidation_state))
        minos.append(float(element.min_oxidation_state))
        amass.append(float(str(element.atomic_mass).split("a")[0]))
####    print(element.atomic_radius)
####    aradi.append(float(str(element.atomic_radius).split("a")[0]))
#       mende.append(float(element.mendeleev_no))

    features.extend(atomicNo)
    pf1.append(features[:])
    features.extend(natom)
    pf2.append(features[:])
    features.extend(group)
    pf3.append(features[:])
    features.extend(row)
    pf4.append(features[:])
    features.append(spacegroup)
    pf5.append(features[:])
    features.append(pressure)
    pf6.append(features[:])
    features.extend(amass)
    pf7.append(features[:])

sp = cross_val_score(rfr,pf1,tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 1 data: "+ str(round(abs(mean(sp)), 3)) + "  ")
sp = cross_val_score(rfr,pf2,tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 2 data: "+ str(round(abs(mean(sp)), 3)) + "  ")
sp = cross_val_score(rfr,pf3,tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 3 data: "+ str(round(abs(mean(sp)), 3)) + "  ")
sp = cross_val_score(rfr,pf4,tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 4 data: "+ str(round(abs(mean(sp)), 3)) + "  ")
sp = cross_val_score(rfr,pf5,tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 5 data: "+ str(round(abs(mean(sp)), 3)) + "  ")
sp = cross_val_score(rfr,pf6,tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 6 data: "+ str(round(abs(mean(sp)), 3)) + "  ")
sp = cross_val_score(rfr,pf7,tc, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 7 data: "+ str(round(abs(mean(sp)), 3)) + "  ")

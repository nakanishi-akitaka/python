#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pymatgen import Composition, Element
from numpy import zeros, mean
trainFile = open("bandgapDFT.csv","r").readlines()

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
bandgaps = []
naiveFeatures = []

MAX_Z = 100

for line in trainFile:
    split = str.split(line, ',')
    material = Composition(split[0])
    materials.append(material) # KAGAKUSIKI
    naiveFeatures.append(naiveVectorize(material)) # TOKUCHORYO
    bandgaps.append(float(split[1])) # band gap NO YOMIKOMI

baselineError = mean(abs(mean(bandgaps) - bandgaps))
print("Mean Absolute Error : " + str(round(baselineError, 3)) + " eV")




from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model, metrics, ensemble

# sklearn NO random forest KAIKI
rfr = ensemble.RandomForestRegressor(n_estimators=10)

# KOUSA KENSHO SIMASU
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

# sc = cross_val_score(rfr, naiveFeatures, bandgaps, cv=cv, scoring='neg_mean_absolute_error')
# print("MAE by RF with composition data: "+ str(round(abs(mean(sc)), 3)) + " eV")

sc = cross_val_score(rfr, naiveFeatures, bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("MAE", sc, "ave", sc.mean(), "std", sc.std())
# print("MAE by RF with composition data: "+ str(round(abs(mean(sc)), 3)) + " eV")
sc = cross_val_score(rfr, naiveFeatures, bandgaps, cv=cv, scoring='neg_mean_squared_error')
print("MSE", sc, "ave", sc.mean(), "std", sc.std())
# print("MSE by RF with composition data: "+ str(round(abs(mean(sc)), 3)) + " eV")
sc = cross_val_score(rfr, naiveFeatures, bandgaps, cv=cv, scoring='r2')
print("R2 ", sc, "ave", sc.mean(), "std", sc.std())
# print("R2  by RF with composition data: "+ str(round(abs(mean(sc)), 3)) + " eV")


pf1= []
pf2= []
pf3= []
pf4= []
pf5= []
pf6= []
pf7= []
pf8= []

for material in materials:
    theseFeatures = []
    fraction = []
    atomicNo = []
    natom = []
    eneg = []
    group = []
    row = []
    maxos = []
    minos = []

    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
#       print(material,natom,element)
        atomicNo.append(float(element.Z))
        eneg.append(element.X)
        group.append(float(element.group))
        row.append(float(element.row))
        maxos.append(float(element.max_oxidation_state))
        minos.append(float(element.min_oxidation_state))

    theseFeatures.extend(atomicNo)
    pf1.append(theseFeatures[:])
    theseFeatures.extend(natom)
    pf2.append(theseFeatures[:])
    theseFeatures.extend(group)
    pf3.append(theseFeatures[:])
    theseFeatures.extend(row)
    pf4.append(theseFeatures[:])

sp = cross_val_score(rfr, pf1, bandgaps, cv=cv, scoring='r2')
print("R2  by RF with physical 1 data: "+ str(round(abs(mean(sp)), 3)) + "   ")
sp = cross_val_score(rfr, pf2, bandgaps, cv=cv, scoring='r2')
print("R2  by RF with physical 2 data: "+ str(round(abs(mean(sp)), 3)) + "   ")
sp = cross_val_score(rfr, pf3, bandgaps, cv=cv, scoring='r2')
print("R2  by RF with physical 3 data: "+ str(round(abs(mean(sp)), 3)) + "   ")
sp = cross_val_score(rfr, pf4, bandgaps, cv=cv, scoring='r2')
print("R2  by RF with physical 4 data: "+ str(round(abs(mean(sp)), 3)) + "   ")

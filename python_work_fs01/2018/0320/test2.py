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

sc = cross_val_score(rfr, naiveFeatures, bandgaps, cv=cv, scoring='r2')
print("R2  by RF with composition data: "+ str(round(abs(mean(sc)), 3)) + "   ")


pf= [[],[],[],[],[]]

for material in materials:
    theseFeatures = []
    feature = [[],[],[],[],[]]

    for element in material:
        feature[ 1].append(float(element.Z))
        feature[ 2].append(material.get_atomic_fraction(element)*material.num_atoms)
        feature[ 3].append(float(element.group))
        feature[ 4].append(float(element.row))
    for i in range(1,5):
        theseFeatures.extend(feature[i])
        pf[i].append(theseFeatures[:])

for i in range(1,5):
    sc = cross_val_score(rfr, pf[i], bandgaps, cv=cv, scoring='r2')
    print("R2  by RF with physical ", i, " data: ave ", round(sc.mean(), 3)," std ", round(sc.std(), 3))

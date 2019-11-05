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

scores_composition = cross_val_score(rfr, naiveFeatures,\
    bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("Mean Absolute Error by Random Forest with composition data: "\
 + str(round(abs(mean(scores_composition)), 3)) + " eV")




physicalFeatures1= []
physicalFeatures2= []
physicalFeatures3= []
physicalFeatures4= []
physicalFeatures5= []
physicalFeatures6= []
physicalFeatures7= []
physicalFeatures8= []

for material in materials:
    theseFeatures = []
    fraction = []
    atomicNo = []
    eneg = []
    group = []
    row = []
    maxos = []
    minos = []

    for element in material:
        fraction.append(material.get_atomic_fraction(element))
        atomicNo.append(float(element.Z))
        eneg.append(element.X)
        group.append(float(element.group))
        row.append(float(element.row))
        maxos.append(float(element.max_oxidation_state))
        minos.append(float(element.min_oxidation_state))

    mustReverse = False
    if fraction [1] > fraction [0]:
            mustReverse = True

    for features in [fraction, atomicNo, eneg, group, row, maxos, minos]:
        if mustReverse:
            features.reverse()
    theseFeatures.append(fraction[0] / fraction [1])
    physicalFeatures1.append(theseFeatures[:])
    theseFeatures.append(eneg[0] - eneg[1])
    physicalFeatures2.append(theseFeatures[:])
    theseFeatures.extend(group)
    physicalFeatures3.append(theseFeatures[:])
    theseFeatures.extend(row)
    physicalFeatures4.append(theseFeatures[:])
    theseFeatures.extend(maxos)
    physicalFeatures5.append(theseFeatures[:])
    theseFeatures.extend(minos)
    physicalFeatures6.append(theseFeatures[:])

scores_physical = cross_val_score(rfr, physicalFeatures1,\
    bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("Mean Absolute Error by Random Forest with physical 1 data: "\
 + str(round(abs(mean(scores_physical)), 3)) + " eV")
scores_physical = cross_val_score(rfr, physicalFeatures2,\
    bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("Mean Absolute Error by Random Forest with physical 2 data: "\
 + str(round(abs(mean(scores_physical)), 3)) + " eV")
scores_physical = cross_val_score(rfr, physicalFeatures3,\
    bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("Mean Absolute Error by Random Forest with physical 3 data: "\
 + str(round(abs(mean(scores_physical)), 3)) + " eV")
scores_physical = cross_val_score(rfr, physicalFeatures4,\
    bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("Mean Absolute Error by Random Forest with physical 4 data: "\
 + str(round(abs(mean(scores_physical)), 3)) + " eV")
scores_physical = cross_val_score(rfr, physicalFeatures5,\
    bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("Mean Absolute Error by Random Forest with physical 5 data: "\
 + str(round(abs(mean(scores_physical)), 3)) + " eV")
scores_physical = cross_val_score(rfr, physicalFeatures6,\
    bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("Mean Absolute Error by Random Forest with physical 6 data: "\
 + str(round(abs(mean(scores_physical)), 3)) + " eV")

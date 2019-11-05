#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pymatgen import Composition, Element
from numpy import zeros, mean
trainFile = open("bandgapDFT.csv","r").readlines()
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
print("MAE: " + str(round(baselineError, 3)) + " eV")




from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model, metrics, ensemble

# sklearn NO random forest KAIKI
rfr = ensemble.RandomForestRegressor(n_estimators=10)

# KOUSA KENSHO SIMASU
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

scores_composition = cross_val_score(rfr, naiveFeatures,\
    bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with composition data: "\
 + str(round(abs(mean(scores_composition)), 3)) + " eV")




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
    amass = []
    aradi = []
    mende = []

    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
#       print(material,natom,element)
        atomicNo.append(float(element.Z))
        group.append(float(element.group))
        row.append(float(element.row))
        eneg.append(element.X)
        maxos.append(float(element.max_oxidation_state))
        minos.append(float(element.min_oxidation_state))
        amass.append(float(str(element.atomic_mass).split("a")[0]))
#       print(element.atomic_radius)
#       aradi.append(float(str(element.atomic_radius).split("a")[0]))
        mende.append(float(element.mendeleev_no))

    theseFeatures.extend(atomicNo)
    pf1.append(theseFeatures[:])
    theseFeatures.extend(natom)
    pf2.append(theseFeatures[:])
    theseFeatures.extend(group)
    pf3.append(theseFeatures[:])
    theseFeatures.extend(row)
    pf4.append(theseFeatures[:])
    theseFeatures.extend(eneg)
    pf5.append(theseFeatures[:])
    theseFeatures.extend(maxos)
    pf6.append(theseFeatures[:])
    theseFeatures.extend(minos)
    pf7.append(theseFeatures[:])
    theseFeatures.extend(amass)
    pf8.append(theseFeatures[:])
    theseFeatures.extend(mende)
    pf9.append(theseFeatures[:])


sp = cross_val_score(rfr,pf1,bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 1 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf2,bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 2 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf3,bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 3 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf4,bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 4 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf5,bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 5 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf6,bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 6 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf7,bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 7 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf8,bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 8 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf9,bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 9 data: "+ str(round(abs(mean(sp)), 3)) + " eV")

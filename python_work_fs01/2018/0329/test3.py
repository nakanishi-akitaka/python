#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
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
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree         import DecisionTreeRegressor
from sklearn.ensemble     import RandomForestRegressor
from sklearn              import neural_network

# sklearn NO random forest KAIKI
lr  = LinearRegression()
dtr = DecisionTreeRegressor()
rfr = RandomForestRegressor()
nn  = neural_network.MLPRegressor(max_iter=1000)
estimators = {'LR ':lr,'DTR':dtr,'RFR':rfr,'NN ':nn}
# for k, v in estimators.items():
#     print(k,v,type(k),type(v))

# KOUSA KENSHO SIMASU
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
for k,v in estimators.items():
    sc = cross_val_score( v, naiveFeatures, bandgaps, cv=cv, scoring='r2')
    print("R2 by "+k+" with composition data: "+ str(round(abs(mean(sc)), 3)) + "   ")


pf= [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
npf=4
npf+=1
for material in materials:
    theseFeatures = []
    feature = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

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

    for i in range(1,npf):
        theseFeatures.extend(feature[i])
        pf[i].append(theseFeatures[:])

for k,v in estimators.items():
    for i in range(1,npf):
        sc = cross_val_score( v, pf[i], bandgaps, cv=cv, scoring='r2')
        print("R2 by "+k+" with physical ", i, " data: ave ", round(sc.mean(), 3)," std ", round(sc.std(), 3))

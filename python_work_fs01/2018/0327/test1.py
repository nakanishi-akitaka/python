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
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree         import DecisionTreeRegressor
from sklearn.ensemble     import RandomForestRegressor
from sklearn              import neural_network

# sklearn NO random forest KAIKI
lr  = LinearRegression()
dtr = DecisionTreeRegressor()
rfr = RandomForestRegressor()
nn  = neural_network.MLPRegressor()

# KOUSA KENSHO SIMASU
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

sc = cross_val_score( lr, naiveFeatures, bandgaps, cv=cv, scoring='r2')
print("R2  by LR with composition data: "+ str(round(abs(mean(sc)), 3)) + "   ")
sc = cross_val_score(dtr, naiveFeatures, bandgaps, cv=cv, scoring='r2')
print("R2  by DT with composition data: "+ str(round(abs(mean(sc)), 3)) + "   ")
sc = cross_val_score(rfr, naiveFeatures, bandgaps, cv=cv, scoring='r2')
print("R2  by RF with composition data: "+ str(round(abs(mean(sc)), 3)) + "   ")
sc = cross_val_score( nn, naiveFeatures, bandgaps, cv=cv, scoring='r2')
print("R2  by NN with composition data: "+ str(round(abs(mean(sc)), 3)) + "   ")

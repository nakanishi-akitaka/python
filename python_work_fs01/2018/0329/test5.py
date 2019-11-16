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
from sklearn.linear_model   import LinearRegression
from sklearn.linear_model   import OrthogonalMatchingPursuit
from sklearn.linear_model   import RANSACRegressor
from sklearn.linear_model   import TheilSenRegressor
from sklearn.linear_model   import BayesianRidge
from sklearn.tree           import DecisionTreeRegressor
from sklearn.ensemble       import RandomForestRegressor
from sklearn.ensemble       import RandomTreesEmbedding
from sklearn.neural_network import MLPRegressor
from sklearn.mixture        import BayesianGaussianMixture
from sklearn.neighbors      import KNeighborsRegressor
from sklearn.neighbors      import RadiusNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes    import GaussianNB
from sklearn.naive_bayes    import MultinomialNB

# sklearn NO random forest KAIKI
lr  = LinearRegression()
dtr = DecisionTreeRegressor()
rfr = RandomForestRegressor()
rte = RandomTreesEmbedding()
mr  = MLPRegressor(max_iter=1000)
omp = OrthogonalMatchingPursuit()
rr  = RANSACRegressor()
tsr = TheilSenRegressor()
br  = BayesianRidge(n_iter=300,tol=0.001)
bgm = BayesianGaussianMixture()
knr = KNeighborsRegressor(n_neighbors=5)
rnr = RadiusNeighborsRegressor(radius=1.0)
pr  = PLSRegression()
gnb = GaussianNB()
mnb = MultinomialNB()
# estimators = {'LR ':lr,'DTR':dtr,'RFR':rfr,'MR ':mr}
estimators = {'LR ':lr,'DTR':dtr,'RFR':rfr,'OMP':omp,'RR ':rr, 'BR ':br,'BGM':bgm ,'KNR':knr,'RNR':rnr,'PR ':pr}

# KOUSA KENSHO SIMASU
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
for k,v in estimators.items():
    sc = cross_val_score( v, naiveFeatures, bandgaps, cv=cv, scoring='r2')
    print("R2 by "+k+" with composition data: "+ str(round(abs(mean(sc)), 3)) + "   ")

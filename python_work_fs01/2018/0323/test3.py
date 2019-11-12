#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import sqrt
from pymatgen import Composition, Element
from numpy import zeros, mean
trainFile = open("lambda.csv","r").readlines()

materials = []
tc        = []

for line in trainFile:
    split = str.split(line, ',')
    material = Composition(split[0])
    materials.append(material)
    tc.append(float(split[8]))

# error = mean(abs(mean(tc) - tc))
# print("MAE:  " + str(round(error, 3)) + " K")



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
    mass     = float(split[2])
    hrate    = float(split[3])
    pressure = float(split[4])
    lambda0  = float(split[5])
    omegalog = float(split[6])
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
        atomicNo.append(float(element.Z))
        group.append(float(element.group))
        row.append(float(element.row))
#       eneg.append(element.X)
        maxos.append(float(element.max_oxidation_state))
        minos.append(float(element.min_oxidation_state))
        amass.append(float(str(element.atomic_mass).split("a")[0]))
#       mende.append(float(element.mendeleev_no))
    features = []
    features.extend(natom)
    pf1.append(features[:])
    features = []
    features.extend(atomicNo)
    pf2.append(features[:])
    features = []
    features.extend(group)
    pf3.append(features[:])
    features = []
    features.extend(row)
    pf4.append(features[:])
    features = []
    features.append(pressure)
    pf5.append(features[:])
#   features = []
#   features.append(mass)
#   pf7.append(features[:])
#   features = []
#   features.append(hrate)
#   pf8.append(features[:])
#   features = []
#   features.append(lambda0)
#   pf9.append(features[:])
    features = []
    features.extend(amass)
    pf6.append(features[:])
    x=amass[0]*natom[0]+amass[1]*natom[1]
    features = []
    features.append(x)
    pf7.append(features[:])
    features = []
    features.append(1/sqrt(x))
    pf8.append(features[:])
    features = []
    features.append(mass)
    pf9.append(features[:])
    features = []
    features.append(1/sqrt(mass))
    pf10.append(features[:])


for i in range(1,4):
    sp1  = cross_val_score(rfr,pf1, tc, cv=cv, scoring='r2')
    sp2  = cross_val_score(rfr,pf2, tc, cv=cv, scoring='r2')
    sp3  = cross_val_score(rfr,pf3, tc, cv=cv, scoring='r2')
    sp4  = cross_val_score(rfr,pf4, tc, cv=cv, scoring='r2')
    sp5  = cross_val_score(rfr,pf5, tc, cv=cv, scoring='r2')
    sp6  = cross_val_score(rfr,pf6, tc, cv=cv, scoring='r2')
    sp7  = cross_val_score(rfr,pf7, tc, cv=cv, scoring='r2')
    sp8  = cross_val_score(rfr,pf8, tc, cv=cv, scoring='r2')
    sp9  = cross_val_score(rfr,pf9, tc, cv=cv, scoring='r2')
    sp10 = cross_val_score(rfr,pf10,tc, cv=cv, scoring='r2')
    print('Ave. of R2 of CVS of RF: %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f' \
    % (mean(sp1),mean(sp2),mean(sp3),mean(sp4),mean(sp5),mean(sp6),mean(sp7),mean(sp8),mean(sp9),mean(sp10)))
#   print('Std. of R2 of CVS of RF: %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f' \
#   % (sp1.std(),sp2.std(),sp3.std(),sp4.std(),sp5.std(),sp6.std(),sp7.std(),sp8.std(),sp9.std(),sp10.std()))

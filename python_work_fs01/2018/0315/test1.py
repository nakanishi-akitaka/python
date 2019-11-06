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
pf20= []
pf21= []
pf22= []
pf23= []
pf24= []

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
    mende = []
    meltp = []
    volum = []
    therm = []
    noble = []
    trans = []
    earth = []
    metal = []
    alka1 = []
    alka2 = []
    halog = []
    chalc = []
    lanth = []
    actin = []

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
#       aradi.append(float(str(element.atomic_radius).split("a")[0]))
        mende.append(float(element.mendeleev_no))
#       resis.append(float(str(element.electrical_resistivity).split("m")[0]))
#       boilp.append(float(str(element.boiling_point).split("K")[0]))
        meltp.append(float(str(element.melting_point).split("K")[0]))
#       lqrng.append(float(str(element.liquid_range).split("K")[0]))
        volum.append(float(str(element.molar_volume).split("c")[0]))
        therm.append(float(str(element.thermal_conductivity).split("W")[0]))
        noble.append(element.is_noble_gas)
        trans.append(element.is_transition_metal)
        earth.append(element.is_rare_earth_metal)
        metal.append(element.is_metalloid)
        alka1.append(element.is_alkali)
        alka2.append(element.is_alkaline)
        halog.append(element.is_halogen)
        chalc.append(element.is_chalcogen)
        lanth.append(element.is_lanthanoid)
        actin.append(element.is_actinoid)
#   print(Element(x),Element(x).oxidation_states)
#   print(Element(x),Element(x).common_oxidation_states)
#   print(Element(x),Element(x).full_electronic_structure)
#   print(Element(x),Element(x).electronic_structure)                    # html
#   print(Element(x),Element(x).atomic_orbitals)                         # html
#   print(Element(x),Element(x).average_ionic_radius) 
#   print(Element(x),Element(x).ionic_radii)
#   print(Element(x),Element(x).icsd_oxidation_states)     # dir

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
    theseFeatures.extend(meltp)
    pf10.append(theseFeatures[:])
    theseFeatures.extend(volum)
    pf11.append(theseFeatures[:])
    theseFeatures.extend(therm)
    pf12.append(theseFeatures[:])
    theseFeatures.extend(noble)
    pf13.append(theseFeatures[:])
    theseFeatures.extend(noble)
    pf14.append(theseFeatures[:])
    theseFeatures.extend(trans)
    pf15.append(theseFeatures[:])
    theseFeatures.extend(earth)
    pf16.append(theseFeatures[:])
    theseFeatures.extend(metal)
    pf17.append(theseFeatures[:])
    theseFeatures.extend(alka1)
    pf18.append(theseFeatures[:])
    theseFeatures.extend(alka2)
    pf19.append(theseFeatures[:])
    theseFeatures.extend(halog)
    pf20.append(theseFeatures[:])
    theseFeatures.extend(chalc)
    pf21.append(theseFeatures[:])
    theseFeatures.extend(lanth)
    pf22.append(theseFeatures[:])
    theseFeatures.extend(actin)


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
sp = cross_val_score(rfr,pf10,bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 10 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf11,bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 11 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf12,bandgaps, cv=cv, scoring='neg_mean_absolute_error')
print("MAE by RF with physical 12 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf13,bandgaps, cv=cv, scoring='neg_mean_absolute_error') 
print("MAE by RF with physical 13 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf14,bandgaps, cv=cv, scoring='neg_mean_absolute_error') 
print("MAE by RF with physical 14 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf15,bandgaps, cv=cv, scoring='neg_mean_absolute_error') 
print("MAE by RF with physical 15 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf16,bandgaps, cv=cv, scoring='neg_mean_absolute_error') 
print("MAE by RF with physical 16 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf17,bandgaps, cv=cv, scoring='neg_mean_absolute_error') 
print("MAE by RF with physical 17 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf18,bandgaps, cv=cv, scoring='neg_mean_absolute_error') 
print("MAE by RF with physical 18 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf19,bandgaps, cv=cv, scoring='neg_mean_absolute_error') 
print("MAE by RF with physical 19 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf20,bandgaps, cv=cv, scoring='neg_mean_absolute_error') 
print("MAE by RF with physical 20 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf21,bandgaps, cv=cv, scoring='neg_mean_absolute_error') 
print("MAE by RF with physical 21 data: "+ str(round(abs(mean(sp)), 3)) + " eV")
sp = cross_val_score(rfr,pf22,bandgaps, cv=cv, scoring='neg_mean_absolute_error') 
print("MAE by RF with physical 22 data: "+ str(round(abs(mean(sp)), 3)) + " eV")

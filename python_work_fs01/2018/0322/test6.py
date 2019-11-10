#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy  import array
from sklearn.datasets import load_boston
from pymatgen import Composition, Element
from numpy import zeros, mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model, metrics, ensemble

materials = []
xatom=Element("H")
for i in range(3,10):
    if(not xatom.from_Z(i).is_noble_gas):
        for iatom1 in range(1,5):
            for iatom2 in range(1,2):
#               print('%s%.1i%s%.1i' % (xatom.from_Z(i).symbol,iatom1,xatom.symbol,iatom2))
                str_mat=str(xatom.from_Z(i).symbol)+str(iatom1)+str(xatom.symbol)+str(iatom2)
                materials.append(str_mat)
print(materials)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pymatgen import MPRester
import itertools

API_KEY='wuoepAHZnBtMv4gC'

# all_symbols = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"]
all_symbols = ["H","He","Li","Be"]
allBinaries = itertools.combinations(all_symbols, 2)
# print(allBinaries)
# print(type(allBinaries))
# print(list(allBinaries))

with MPRester(API_KEY) as m:
     for system in allBinaries:
         results = m.get_data(system[0]+'-'+system[1],data_type='vasp')
#        print(system)
#        print(system[0]+'-'+system[1])
#        print(results)
         for material in results:
             if material['e_above_hull'] < 1e-6:
                 print(material['pretty_formula']+','+str(material['band_gap']))

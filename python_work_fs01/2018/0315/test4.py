#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pymatgen import MPRester, periodic_table
import itertools

API_KEY='wuoepAHZnBtMv4gC'

# allBinaries = itertools.combinations(periodic_table.all_symbols(), 2)
allBinaries = ["VH2", "V2H", "CrH", "NiH"]

with MPRester(API_KEY) as m:
     for system in allBinaries:
         results = m.get_data(system[0]+'-'+system[1],data_type='vasp')
         for material in results:
             if material['e_above_hull'] < 1e-6:
                 print(material['pretty_formula']+','+str(material['band_gap']))

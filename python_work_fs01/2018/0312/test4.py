#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from pymatgen import Lattice
from pymatgen import IStructure
a=4.209
latt=Lattice.cubic(a)
structure=IStructure(latt,["Cs","Cl"],[[0,0,0],[0.5,0.5,0.5]])
print(structure.density)
print(structure.distance_matrix)
print(structure.get_distance)

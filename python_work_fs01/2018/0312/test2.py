#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from pymatgen import Element
from pymatgen import Composition
comp=Composition("LiFePO4")
print(comp.num_atoms)
print(comp.formula)
print(comp.get_atomic_fraction(Element("Li")))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from pymatgen import Lattice
l=Lattice([1,0,0,0,1,0,0,0,1])
print(l._angles)
print(l.is_orthogonal)
print(l._lengths)
print(l._matrix)

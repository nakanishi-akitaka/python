#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from pymatgen import MPRester
from pymatgen.electronic_structure.plotter import BSPlotter
a=MPRester("wuoepAHZnBtMv4gC")
bs=a.get_bandstructure_by_material_id("mp-3748")
print(bs.is_metal())
print(bs.get_band_gap())
print(bs.get_direct_band_gap())
plotter=BSPlotter(bs)
plotter.get_plot().show()

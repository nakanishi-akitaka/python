#!/usr/bin/env python
import glob
ls = glob.glob("sample/sample2/*.out")
energy=0.0
a=10.0
energy_min=0.0
a_min=0.0
for name in ls:
    f=open(name,"r")
    for line in f:
        if "total energy" in line:
            data = line.split()
            energy = float(data[3])
        if "lattice" in line:
            data = line.split()
            a = float(data[4])
        if energy_min > energy:
            energy_min = energy 
            a_min = a
    f.close()
print "%f %f"  % (a_min, energy_min)

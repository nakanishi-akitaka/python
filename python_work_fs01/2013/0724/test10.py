#!/usr/bin/env python
import glob
ls = glob.glob("sample/sample2/*.out")
for name in ls:
    f=open(name,"r")
    for line in f:
        if "total energy" in line:
            data = line.split()
            energy = float(data[3])
        if "lattice" in line:
            data = line.split()
            a = float(data[4])
    f.close()
    print "%f %f"  % (a, energy)

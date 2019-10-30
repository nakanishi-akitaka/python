#!/usr/bin/env python
f=open("sample/sample2/file00.out","r")
for line in f:
    if "total energy" in line:
        data = line.split()
        energy = float(data[3])
    if "lattice" in line:
        data = line.split()
        a = float(data[4])
f.close()
print "%f %f"  % (a, energy)

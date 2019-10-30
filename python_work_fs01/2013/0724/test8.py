#!/usr/bin/env python
import glob
ls = glob.glob("sample/sample1/*.kkr")
for name in ls:
    f=open(name,"r")
    for line in f:
        if "total energy" in line:
            data = line.split("=")
            energy = float(data[1])
        if "bcc" in line:
            data = line.split()
            a = float(data[2])
    f.close()
    print "%f %f"  % (a, energy)

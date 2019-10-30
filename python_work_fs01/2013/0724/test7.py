#!/usr/bin/env python
f=open("sample/sample1/file00.kkr","r")
for line in f:
    if "total energy" in line:
        data = line.split("=")
        energy = float(data[1])
    if "bcc" in line:
        data = line.split()
        a = float(data[2])
f.close()
print "%f %f"  % (a, energy)

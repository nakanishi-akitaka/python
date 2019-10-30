#!/usr/bin/env python
f=open("sample/sample1/file00.kkr","r")
for line in f:
    if "total energy" in line:
        data = line.split("=")
        # energy = float(data[1])
        energy = data[1]
f.close()
print energy

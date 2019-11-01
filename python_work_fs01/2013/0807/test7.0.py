#!/usr/bin/env python

atom1={"H":1, "He":2}
atom2={"Li":3, "Be":4}

for (x,y) in atom2.items():
    atom1[x]=y
print atom1

#!/usr/bin/env python

def addDict(d1,d2):
    s={}
    for (x,y) in d1.items():
        s[x]=y
    for (x,y) in d2.items():
        s[x]=y
    return s

atom1={"H":1, "He":2}
atom2={"Li":3, "Be":4}
print atom1
print atom2
print addDict(atom1,atom2)


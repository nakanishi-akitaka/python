#!/usr/bin/env python
atom={"H":1, "He":2, "Li":3, "Be":5}

for x in atom:
    print x, atom[x]

for (x,y) in atom.items():
    print x,y

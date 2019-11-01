#!/usr/bin/env python

atom={"H":1, "He":2, "Li":3, "Be":4}
print atom
print atom["Li"]
print atom.items()
print atom.items()[2]
print atom.keys()
print atom.keys()[2]
print atom.values()
print atom.values()[2]
atom.clear()
print atom

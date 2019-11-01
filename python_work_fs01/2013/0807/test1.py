#!/usr/bin/env python
atom={"H":1, "He":2, "Li":3, "Be":5}
print atom["Be"]
atom["Be"]=4
print atom["Be"]
if "He" in atom:
    print "He is atom"

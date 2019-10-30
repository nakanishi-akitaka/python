#!/usr/bin/env python
f=open("sample/sample1/file00.kkr","r")
for line in f:
    if "total energy" in line:
        print line
f.close()

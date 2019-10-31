#!/usr/bin/env python
import re
f=open("sample/sample2/file00.out","r")
for line in f:
    if re.search(r"[tT]otal [eE]nergy",line):
        print line
f.close()

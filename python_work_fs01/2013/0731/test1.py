#!/usr/bin/env python
text1="total energy=-800.0"
text2="Total Energy=-800.0"
print "total energy" in text1
print "total energy" in text2
import re
print re.search(r"(t|T)otal (e|E)nergy",text1)
print re.search(r"(t|T)otal (e|E)nergy",text2)

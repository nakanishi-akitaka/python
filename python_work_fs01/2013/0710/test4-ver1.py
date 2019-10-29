#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
f=open("cnt.xyz","w")
n=4
f.write("%d\n" % (n*n*2))
f.write("CNT Structure\n")

# a = ( 1, 0), b = ( -1/2, sqrt(3)/2)
# C=na+mb
# T=ua+vb
# atomic position
# tau1 = (0, 0)
# tau2 = ( 0, sqrt(3)/3) =r

l=1.44*math.sqrt(3.0)
a=[ 1.0*l, 0.0]
b=[-0.5*l, math.sqrt(3.0)/2.0*l]
r=[ 0.0  , math.sqrt(3.0)/3.0*l]
# print a, a[0], a[1]
# print b, b[0], b[1]
# print r, r[0], r[1]
for i in xrange(0,n):
    for j in xrange(0,n):
        x=i*a[0]+j*b[0]
        y=i*a[1]+j*b[1]
        z=0.0
        f.write("C %f %f %f\n" % (x,y,z))
        x=x+r[0]
        y=y+r[1]
        z=0.0
        f.write("C %f %f %f\n" % (x,y,z))
f.close()

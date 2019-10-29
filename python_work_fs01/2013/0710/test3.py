#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
f=open("graphene.xyz","w")
n=4
f.write("%d\n" % (n*n*2))
f.write("Hexagonal Structure\n")

for i in xrange(0,n):
    for j in xrange(0,n):
        x=1.42*i-1.42/2.0*j
        y=1.42*math.sqrt(3.0)/2.0*j
        z=0.0
        f.write("C %f %f %f\n" % (x,y,z))
        y+=1.42*math.sqrt(3.0)/2.0*2.0/3.0
        f.write("C %f %f %f\n" % (x,y,z))
f.close()

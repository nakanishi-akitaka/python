#!/usr/bin/env python
# -*- coding: utf-8 -*-

f=open("nacl.xyz","w")
n=4
f.write("%d\n" % (n*n*n))
f.write("NaCl Structure\n")

for i in xrange(0,n):
    for j in xrange(0,n):
        for k in xrange(0,n):
            x=5.63*i
            y=5.63*j
            z=5.63*k
            if (i+j+k)%2 ==0:
                f.write("Na %f %f %f\n" % (x,y,z))
            else:
                f.write("Cl %f %f %f\n" % (x,y,z))
f.close()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

def f(n):
    s=0
    for i in xrange(1,n):
        if i % 2 == 0:
            s += i
    return s

f=open("test1.out","w")
f.write("Hello\n")
f.write("World\n")
f.write("%d %d %d" % (1,2,3) )
f.close()

for i in xrange(0, 10):
    a=5.0+i*0.01
    f=open("input%d.out" % i,"w")
    f.write("Unit vector\n")
    f.write("%f 0.0 0.0\n" % a)
    f.write("0.0 %f 0.0\n" % a)
    f.write("0.0 0.0 %f\n" % a)
    f.close()


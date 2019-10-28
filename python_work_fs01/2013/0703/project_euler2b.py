#!/usr/bin/env python
# -*- coding: utf-8 -*-

a=0
print a
b=1
print b
s=0
for i in xrange(1, 40):
    c=a+b
    a=b
    b=c
    if c < 4000000:
        if c % 2 == 0:
            print c
            s += c

print s

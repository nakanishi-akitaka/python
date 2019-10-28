#!/usr/bin/env python
# -*- coding: utf-8 -*-

a=0
print a
b=1
print b
for i in xrange(1, 15):
    c=a+b
    print c
    a=b
    b=c

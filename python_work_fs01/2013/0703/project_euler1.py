#!/usr/bin/env python
# -*- coding: utf-8 -*-

def f(n):
    s=0
    for i in xrange(1,n):
        if i % 3 == 0:
            s += i
        elif i % 5 == 0:
            s += i
    return s

print f(1000)

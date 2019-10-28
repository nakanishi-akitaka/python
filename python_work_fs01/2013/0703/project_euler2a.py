#!/usr/bin/env python
# -*- coding: utf-8 -*-

a=0
b=1
s=0
c=0
while c < 4000000:
    c=a+b
    a=b
    b=c
    if c % 2 == 0:
        s += c

print s

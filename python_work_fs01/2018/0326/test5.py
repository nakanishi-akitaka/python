#!/usr/bin/env python
# -*- coding: utf-8 -*-

# test08-1.py 
# def kasan(x,y):
#     z = x + y
#     return z
# 
# a = kasan(12,34)
# print(a)

# test08-2.py 
# def argtest1(*a):
#     n = len(a)
#     for m in range(n):
#         print(a[m])
#     return n
# a = argtest1('a',1,'b',2)
# print('number of index ',a)

# test08-3.py 
# def argtest2(**ka):
#     print('name:',   ka['name'])
#     print('age:' ,   ka['age'])
#     print('country:',ka['country'])
#     n = len(ka)
#     return n
# 
# a = argtest2(name='tanaka',country='japan', age=41)
# print('number of index ',a)

# test08-3.py 
gv = 'initial value'
def scopetest1():
    global gv
    print('scopetest1 no Naibu:      ', gv)
    gv = 'scopetest1 change gv'

def scopetest2():
    gv = 'scopetest2 local parameter gv'
    print(gv)

print('global parameter gv')
print('scopetest1 before calling:', gv)
scopetest1()
print('scopetest1 after  calling:', gv)
scopetest2()
print('scopetest2 after  calling:', gv)

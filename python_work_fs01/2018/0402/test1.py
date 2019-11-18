#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import sum
import random
from math import exp, sin, pi, sqrt

def area(radius):
    return pi * radius**2
####print(area(5))

####def absolute_value(x):
####    if x < 0:
####        return -x
####    if x > 0:
####        return  x
#####   else:
#####       return x
####print(absolute_value(10))
####print(absolute_value(-10))
####print(absolute_value(0))

####def hikaku(x,y):
####    if x > y:
####        return 1
####    if x < y:
####        return -1
####    if x == y:
####        return 0
####print(hikaku(1,2))
####print(hikaku(1,1))
####print(hikaku(2,1))

def distance(x1,y1,x2,y2):
    dx = x2 - x1
    dy = y2 - y1
    result = sqrt(dx**2+dy**2)
    return result
####print(distance(1,2,4,6))

####def hypotenuse(a,b):
####    result = sqrt(a**2+b**2)
####    return result
####
####print(hypotenuse(3,4))

def circle_area(xc,yc,xp,yp):
    return area(distance(xc,yc,xp,yp))
####    radius = distance(xc,yc,xp,yp)
####    result = area(radius)
####    return result
####print(circle_area(1,2,4,6))

def is_divisible(x,y):
    return x % y == 0
####    if x % y == 0:
####        return True
####    else:
####        return False
####print(is_divisible(6,4))
####print(is_divisible(6,3))
####if is_divisible(6,3):
####    print('6 is divisible by 3')

def is_between(x,y,z):
    if(x <= y and y <= z):
        return True
    else:
        return False
####print(is_between(1,2,3))
####print(is_between(1,3,2))
####print(is_between(2,1,3))
####print(is_between(2,3,1))
####print(is_between(3,1,2))
####print(is_between(3,2,1))

####def factorial(n):
####    if n == 0:
####        return 1
####    else:
####        recurse = factorial(n-1)
####        result = n * recurse
####        return result
####print(factorial(3))

####def factorial(n):
####    if not isinstance(n,int):
####        print('Factroial is only defined for integers.')
####        return None
####    elif n < 0:
####        print('Factroial is not defined for negative integers.')
####        return None
####    elif n == 0:
####        return 1
####    else:
####        recurse = factorial(n-1)
####        result = n * recurse
####        return result
####print(factorial(1.5))
####print(factorial(-2))
####print(factorial(3))

def factorial(n):
    space=' '*(4*n)
    print(space, 'factorial', n)
    if n == 0:
        print(space, 'returning 1')
        return 1
    else:
        recurse = factorial(n-1)
        result = n * recurse
        print(space, 'returning', result)
        return result
print(factorial(5))

def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1)+fibonacci(n-2)
####for i in range(20):
####     print(fibonacci(i))

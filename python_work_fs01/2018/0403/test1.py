#!/usr/bin/env python
# -*- coding: utf-8 -*-

#### section 7.[1-2]
#### nothing

#### section 7.3
def countdown(n):
    while n > 0:
        print(n)
        n -=1
    print('Blast off!')
####countdown(3)

def sequence(n):
    while n != 1:
        print(n)
        if n%2 == 0:
            n = n/2
        else:
            n = n*3+1
####sequence(10)

#### section 7.4
def function74():
    while True:
        line = input('<')
        if line == 'done':
            break
        print(line)
    print('Done!')
####function74()

#### section 7.5
def square_root(a):
    epsilon = 0.0000001
    x = a - epsilon
    while True:
        print(x)
        y = (x+a/x)/2
        if abs(y-x) < epsilon:
            break
        x = y
####square_root(4)

#### section 7.[6-8]
#### nothing

#### section 7.9
#### train 7.3
from math import sqrt

def test_square_root(a):
    epsilon = 0.0000001
    x = a - epsilon
    while True:
        y = (x+a/x)/2
        if abs(y-x) < epsilon:
            return y
        x = y

####for i in range(1,10):
####    x=sqrt(i)
####    y=test_square_root(i)
####    print('%.8f %.8f %.8f' % (x,y,x-y))

#### train 7.4
def train74():
    while True:
        line = input('<')
        if line == 'done':
            break
        print(eval(line))
    print('Done!')
####train74()

#### train 7.5
from math import sqrt, pi, factorial
def estimate_pi():
    epsilon = 1.0e-15
    x=0
    k=0
    while True:
        dx=2.0*sqrt(2.0)*factorial(4*k)*(1103.0+26390.0*float(k))/(99.0**2*(396.0**k*factorial(k))**4)
        if abs(dx) < epsilon:
            return 1.0/x
        x+=dx
        print(k,x)
        k+=1
####print(estimate_pi(),pi)

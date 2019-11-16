#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
sys.path.append('/home/nakanishi/python/lib/swampy-2.1.7/swampy/')
import Tkinter
from swampy.TurtleWorld import *
from math import pi

world = TurtleWorld()
bob = Turtle()
bob.delay = 0.001

def square(t,length):
    for i in range(4):
       fd(t,length)
       lt(t)
# square(bob,100)

def polygon(t,n,length):
    for i in range(n):
        fd(t,length)
        lt(t,360.0/n)
# polygon(bob,60,10)
# polygon(bob,n=6,length=100)
# polygon(bob,63,10)

def circle(t,r):
    n = int(2.0*pi*r/4.0)+1
    length = 2.0*pi*r/n
#   print(n,length,360.0/n)
    polygon(bob,n,length)
circle(bob,50)

def arc(t,r,angle):
    arc_length = 2.0*pi*r*angle/360.0
    n = int(arc_length/4)+1
    step_length = arc_length/n
    step_angle  = float(angle)/n
#   print(n,step_length,step_angle)
    for i in range(n):
        fd(t,step_length)
        lt(t,step_angle)
arc(bob,50,360)


# wait_for_user()

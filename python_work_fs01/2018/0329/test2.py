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
def polyline(t,n,length,angle):
    for i in range(n):
       fd(t,length)
       lt(t,angle)

def polygon(t,n,length):
    polyline(t,n,length,360.0/n)

def arc(t,r,angle):
    arc_length = 2*pi*r*abs(angle)/360
    n=int(arc_length/4)+1
    step_length=arc_length/n
    step_angle =float(angle)/n
    lt(t,step_angle/2)
    polyline(t,n,step_length,step_angle)
    rt(t,step_angle/2)

def circle(t,r):
    arc(t,r,360)

# polygon(bob,63,10)
arc(bob,50,240)
arc(bob,50,240)
arc(bob,50,240)
circle(bob,50)



# wait_for_user()

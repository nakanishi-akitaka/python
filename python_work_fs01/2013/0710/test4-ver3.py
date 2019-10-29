#!/usr/bin/env python
# -*- coding: utf-8 -*-

n=4
m=0
import math

# a = ( 1, 0), b = ( -1/2, sqrt(3)/2)
# C=na+mb
# T=ua+vb
#   (u,v)=(2*m-n, -2n+m)
# atomic position
# x1 = (0, 0)
# x2 = ( 0, sqrt(3)/3)

l = 1.44*math.sqrt(3.0)
a = [ 1.0*l, 0.0]
b = [-0.5*l, math.sqrt(3.0)/2.0*l]
c = [n*a[0]+m*b[0], n*a[1]+m*b[1]]
u =-2*m+n
v = 2*n-m
t = [u*a[0]+v*b[0], u*a[1]+v*b[1]]
norm_c=math.sqrt(c[0]**2+c[1]**2)
norm_t=math.sqrt(t[0]**2+t[1]**2)
print c, t, norm_c, norm_t, c[0]*t[0]+c[1]*t[1]

tmp=[0,n,u,n+u]
min1=min(tmp)
max1=max(tmp)
tmp=[0,m,v,m+v]
min2=min(tmp)
max2=max(tmp)
ncell1=max1-min1
ncell2=max2-min2

f=open("cns_%d_%d.xyz" % (n, m), "w")
f.write("%d\n" % (ncell1*ncell2*2))
f.write("CNS Structure\n")

for i in xrange(min1,max1):
    for j in xrange(min2,max2):
        x=i*a[0]+j*b[0]
        y=i*a[1]+j*b[1]
        z=0.0
        projection_c=(x*c[0]+y*c[1])/norm_c/norm_c
        projection_t=(x*t[0]+y*t[1])/norm_t/norm_t
        if (projection_c >= 0.0 and projection_c <= 1.0) and (projection_t >= 0.0 and projection_t <= 1):
            f.write("C %f %f %f\n" % (x,y,z))
        else: 
            f.write("H %f %f %f\n" % (x,y,z))
        y+=l/math.sqrt(3.0)
        projection_c=(x*c[0]+y*c[1])/norm_c/norm_c
        projection_t=(x*t[0]+y*t[1])/norm_t/norm_t
        if (projection_c >= 0.0 and projection_c <= 1.0) and (projection_t >= 0.0 and projection_t <= 1):
            f.write("C %f %f %f\n" % (x,y,z))
        else: 
            f.write("H %f %f %f\n" % (x,y,z))
f.close()

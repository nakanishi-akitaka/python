#!/usr/bin/env python
# -*- coding: utf-8 -*-

n=8
m=4
e=0.00001 # round-off error
import math

# a = ( 1, 0), b = ( -1/2, sqrt(3)/2)
# C=na+mb
# T=ua+vb
#   (u,v)=(2*m-n, -2n+m)
# atomic position
# x1 = (0, 0)
# x2 = ( 0, sqrt(3)/3)

l = 1.44*math.sqrt(3.0)
ax = 1.0*l
ay = 0.0
bx = -0.5*l
by = math.sqrt(3.0)/2.0*l
cx = n*ax+m*bx
cy = n*ay+m*by
u =-2*m+n
v = 2*n-m
tx = u*ax+v*bx
ty = u*ay+v*by
norm_c=math.sqrt(cx**2+cy**2)
norm_t=math.sqrt(tx**2+ty**2)

tmp=[0,n,u,n+u]
min1=min(tmp)
max1=max(tmp)
tmp=[0,m,v,m+v]
min2=min(tmp)
max2=max(tmp)
ncell1=max1-min1
ncell2=max2-min2

f=open("cns_%d_%d.xyz2" % (n, m), "w")
f.write("%d\n" % (ncell1*ncell2*2))
f.write("CNS Structure\n")

p=0
cnt=[0.0]
for i in xrange(min1,max1):
    for j in xrange(min2,max2):
        x=i*ax+j*bx
        y=i*ay+j*by
        z=0.0
        projection_c=(x*cx+y*cy)/norm_c/norm_c
        projection_t=(x*tx+y*ty)/norm_t/norm_t
        if (projection_c > -e and projection_c <  1+e) and (projection_t > -e and projection_t <  1+e):
            radius=norm_c/2.0/math.pi
            theta=projection_c*2.0*math.pi
            x2=radius*math.cos(theta)
            y2=radius*math.sin(theta)
            z2=projection_t*norm_t
            f.write("C %f %f %f\n" % (x,y,z))
            cnt+=[x2,y2,z2]
            p=p+1
        else: 
            f.write("H %f %f %f\n" % (x,y,z))
        y+=l/math.sqrt(3.0)
        projection_c=(x*cx+y*cy)/norm_c/norm_c
        projection_t=(x*tx+y*ty)/norm_t/norm_t
        if (projection_c > -e and projection_c <  1+e) and (projection_t > -e and projection_t <  1+e):
            radius=norm_c/2.0/math.pi
            theta=projection_c*2.0*math.pi
            x2=radius*math.cos(theta)
            y2=radius*math.sin(theta)
            z2=projection_t*norm_t
            f.write("C %f %f %f\n" % (x,y,z))
            cnt+=[x2,y2,z2]
            p=p+1
        else: 
            f.write("H %f %f %f\n" % (x,y,z))
f.close()

f=open("cnt_%d_%d.xyz2" % (n, m), "w")
f.write("%d\n" % p )
f.write("CNT Structure\n")
for i in xrange(1, p+1):
    x=cnt[3*(i-1)+1]
    y=cnt[3*(i-1)+2]
    z=cnt[3*(i-1)+3]
    f.write("C %f %f %f\n" % (x ,y ,z ))

f.close()
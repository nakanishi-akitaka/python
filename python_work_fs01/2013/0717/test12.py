#!/usr/bin/env python
prime=[2,3,5,7]
n=10000
for i in xrange(11,n):
    l=len(prime)
    flag=0
    for j in xrange(l):
        if i % prime[j] == 0:
            flag=1
    if flag == 0:
        prime+=[i]
print len(prime)
print max(prime)

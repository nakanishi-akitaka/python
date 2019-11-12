#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 2.8.2 lambda to Kansuu Teigi
dbl2=lambda n:2*n
print(dbl2(3))
wa = lambda a,b: a+b
print(wa(2,3))
print((lambda a,b : a+b)(2,3))

# 2.8.3 filter
from random import randrange
lst = list(map(randrange,[100]*17))
print(lst)
lst2 = list(filter(lambda n:n%2==0,lst))
print(lst2)

# 2.8.4
evod = lambda n :'Even' if n%2==0 else 'Odd'
print(evod(2))
print(evod(3))

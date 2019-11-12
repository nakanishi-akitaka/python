#!/usr/bin/env python
# -*- coding: utf-8 -*-

# def dbl(n):
#     return 2*n
# 
# print(dbl(3))
# res = map(dbl,[1,2,3])
# print(res)
# print(list(res))
# print(list(map(dbl,[1,2,3])))

# map01.py
import time                   # Jikan Keisoku
from random import randrange  # Ransuu Hassei

# Guusuu Kisuu wo Hantei suru Kansuu no Teigi
def EvenOrOdd(n):
    if n % 2 == 0:
        return('Even')
    else:
        return('Odd')

# Kokoromi (1)
# Ransuu list no seisei(1): Mijikai mono
lst = list(map(randrange, [100]*10))

# Guusuu Kisuu no Sikibetu Kekka
lst2 = list(map(EvenOrOdd, lst))
print('10 Ko no Ransuu no Guusuu Kisuu no Hantei')
print(lst)
print(lst2,'\n')

# Kokoromi (2)
# Ransuu list no seisei(2): 1,000,000 ko
print('1,000,000 Ko no Ransuu no Guusuu Kisuu no Hantei')
t1 = time.time()
lst = list(map(randrange, [100]*1000000))
t = time.time() - t1
print('Ransuu Seisei ni You sita Jikan:',t,'byou')

# Sokudo Kensa(1): for ni yoru Syori
lst2 = []
t1 = time.time()
for i in range(1000000):
    lst2.append(EvenOrOdd(lst[i]))
t = time.time() - t1
print('for ni Yoru Syori:',t,'byou')

# Sokudo Kensa(2): map ni yoru Syori
t1 = time.time()
lst2 = list(map(EvenOrOdd, lst))
t = time.time() - t1
print('map ni Yoru Syori:',t,'byou')

# Hantei kekka no Kakuninn
# print(lst2)

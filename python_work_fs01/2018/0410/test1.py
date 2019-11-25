#!/usr/bin/env python
# -*- coding: utf-8 -*-

#### section 12.1
# t='a','b','c','d','e'
# print(t)
# t=('a','b','c','d','e')
# print(t)
# t='a',
# print(type(t))
# t=('a')
# print(type(t))
# t=tuple()
# print(type(t))
# t=tuple('lupins')
# print(t)
# t=tuple('abcde')
# print(t[0])
# print(t[1:3])
# t=('A',)+t[1:]
# print(t)

#### section 12.3 (typo 12.2?)
# a=1
# b=99
# print(a,b)
# a,b=b,a
# print(a,b)
# addr='monty@python.org'
# uname, domain = addr.split('@')
# print(uname)
# print(domain)

#### section 12.3
# t=divmod(7,3)
# print(t)
# quot,rem=divmod(7,3)
# print(quot)
# print(rem)
# 
# def min_max(t):
#     return min(t), max(t)
# t=[i for i in range(10)]
# print(t)
# print(min_max(t))

#### section 12.4
# def print_all(*args):
#     print(args)
# print_all(1,2.0,'3')
# t=(7,3)
# print(divmod(*t))

#### train 12.1
# print(max(1,2,3))
# def sum_all(*args):
#     s=[t for t in args]
#     print(sum(s))
# t=(1,2,3)
# print(t)
# sum_all(*t)

#### section 12.5
# s='abc'
# t=[0,1,2]
# for a,b in zip(s,t):
#     print(a,b)
# print(zip(s,t))
# print(list(zip(s,t)))
# s='Anne'
# t='Elk'
# print(list(zip(s,t)))
# t=[('a',0), ('b',1), ('c',2)]
# for letter, number in t:
#     print(letter,number)
# def has_match(t1,t2):
#     for x, y in zip(t1,t2):
#         if x==y:
#             return True
#     return False
# s=[i for i in range(11)]
# t=[i for i in range(10,21)]
# print(has_match(s,t))
# for i, e in enumerate('abc'):
#     print(i,e)

#### section 12.6
# d={'a':0,'b':1,'c':2}
# t=d.items()
# print(type(t))
# print(list(t))
# t=[('a',0), ('b',1), ('c',2)]
# d=dict(t)
# print(d)
# d=dict(zip('abc',range(3)))
# print(d)
# 
# for k, v in d.items():
#     print(v,k)
# 
# d={('a','b'):0,('c','d'):1,('e','f'):2}
# for t1, t2 in d:
#     print(t1,t2,d[t1,t2])
# t1=tuple('ace')
# t2=tuple('bdf')
# t3=zip(t1,t2)
# d=dict(zip(t3,range(3)))
# print(t1)
# print(t2)
# print(t3)
# print(d)

#### section 12.7
print((0,1,2)<(0,3,4))
print((0,1,200000000)<(0,3,4))
def sort_by_length(words):
    t=[]
    for word in words:
        t.append((len(word),word))
    t.sort(reverse=True)
    res=[]
    for length, word in t:
        res.append(word)
    return res
fruits=['banana','ringo','kaki','ithijiku','mango','budo']
print(fruits)
print(sort_by_length(fruits))

#### train 12.2 -> skip

#### section 12.8
# nothing

#### section 12.9
# nothing

#### section 12.10
# nothing

#### section 12.11
#### train 12.3
#### train 12.4
#### train 12.5
#### train 12.6

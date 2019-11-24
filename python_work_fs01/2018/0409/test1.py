#!/usr/bin/env python
# -*- coding: utf-8 -*-

#### section 11.0
# eng2sp=dict()
# print(eng2sp)
# eng2sp['one']='uno'
# print(eng2sp)
# eng2sp={'one':'uno','two':'dos','three':'tres'}
# print(eng2sp)
# print(eng2sp['two'])
# print(len(eng2sp))
# print('one' in eng2sp)
# print('uno' in eng2sp)
# print('uno' in eng2sp.values())

#### train 11.1 -> skip

#### section 11.1
# def histogram(s):
#     d=dict()
#     for c in s:
#         if c not in d:
#             d[c]=1
#         else:
#             d[c]+=1
#     return d
# print(histogram('brontosaurus'))

#### train 11.2 
# h=histogram('a')
# print(h.get('a',0))
# print(h.get('b',0))
# def histogram2(s):
#     d=dict()
#     for c in s:
#         d[c]=d.get(c,0)+1
#     return d
# print(histogram2('brontosaurus'))

#### section 11.2
# def print_hist(h):
#     for c in h:
#         print(c,h[c])
# h=histogram2('parrot')
# print_hist(h)

#### train 11.3
# def histogram2(s):
#     d=dict()
#     for c in s:
#         d[c]=d.get(c,0)+1
#     return d
# def print_hist(h):
#     t=list(h.keys())
#     t.sort()
#     for c in t:
#         print(c,h[c])
# h=histogram2('parrot')
# print_hist(h)

#### section 11.3
# def histogram2(s):
#     d=dict()
#     for c in s:
#         d[c]=d.get(c,0)+1
#     return d
# def reverse_lookup(d,v):
#     for k in d:
#         if d[k]==v:
#             return k
#     raise ValueError
# 
# h=histogram2('parrot')
# print(reverse_lookup(h,2))
# print(reverse_lookup(h,3))

#### train 11.4 -> skip

#### section 11.4
# def histogram2(s):
#     d=dict()
#     for c in s:
#         d[c]=d.get(c,0)+1
#     return d
# def invert_dic(d):
#     inverse = dict()
#     for key in d:
#         val = d[key]
#         if val not in inverse:
#             inverse[val] = [key]
#         else:
#             inverse[val].append(key)
#     return inverse
# 
# h=histogram2('parrot')
# print(h)
# print(invert_dic(h))

#### train 11.3 (typo5?) -> skip

#### section 11.5
# known = {0:0,1:1}
# def fibonacci(n):
#     if n in known:
#         return known[n]
#     res = fibonacci(n-1)+fibonacci(n-2)
#     known[n]=res
#     return res
# 
# for i in range(10):
#     print(fibonacci(i))

#### train 11.6 -> skip

#### train 11.7 -> skip

#### section 11.6
# verbose = True
# def example1():
#     if verbose:
#         print('Running example1')
# example1()
# 
# been_called=False
# def example2():
#     global been_called
#     been_called=True
# example2()
# print(been_called)
# 
# count=0
# def example3():
#     global count
#     count+=1
# example3()
# 
# known={0:0,1:1}
# def example4():
#     known[2] = 1
# example4()
# print(known)
# 
# def example5():
#     global known
#     known=dict()
# example5()
# print(known)

#### section 11.7
known = {0:0,1:1}
def fibonacci(n):
    if n in known:
        return known[n]
    res = fibonacci(n-1)+fibonacci(n-2)
    known[n]=res
    return res
print(fibonacci(50))

#### train 11.8 -> skip

#### section 11.8 
# nothing

#### section 11.9
# nothing

#### section 11.10
#### train 11.9  -> skip
#### train 11.10 -> skip
#### train 11.11 -> skip

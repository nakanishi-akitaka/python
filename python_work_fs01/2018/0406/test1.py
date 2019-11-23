#!/usr/bin/env python
# -*- coding: utf-8 -*-

#### section 10.1
# cheeses=['Cheddar','Edam','Gouda']
# numbers=[17,123]
# empty=[]
# print(cheeses,numbers,empty)

#### section 12.2 (typo 10.2)
# cheeses=['Cheddar','Edam','Gouda']
# print(cheeses[0])
# numbers=[17,123]
# numbers[1]=5
# print(numbers)
# print('Edam' in cheeses)
# print('Brie' in cheeses)

#### section 10.3
# cheeses=['Cheddar','Edam','Gouda']
# for cheese in cheeses:
#     print(cheese)
# numbers=[17,123]
# for i in range(len(numbers)):
#     numbers[i]=numbers[i]*2
# print(numbers)
# for x in []:
#     print('This never happens.')

#### section 10.4
# a=[1,2,3]
# b=[4,5,6]
# c=a+b
# print(c)
# print([0]*4)
# print([1,2,3]*4)

#### section 10.5
# t=['a','b','c','d','e','f']
# print(t[1:3])
# print(t[:4])
# print(t[3:])
# print(t[:])
# t[1:3]=['x','y']
# print(t)

#### section 10.6
# t=['a','b','c']
# t.append('d')
# print(t)
# t1=['a','b','c']
# t2=['d','e']
# t1.extend(t2)
# print(t1)
# t=['d','c','b','a']
# t.sort()
# print(t)

#### section 10.7
# def add_all(t):
#     total=0
#     for x in t:
#         total+=x
#     return total
# t=[1,2,3]
# print(add_all(t))
# print(sum(t))

#### train 10.1 -> skip
# def capitalize_all(t):
#     res=[]
#     for s in t:
#         res.append(s.capitalize())
#     return res
# t=['a','b','c']
# print(capitalize_all(t))

#### train 10.2 -> skip
# def only_upper(t):
#     res=[]
#     for s in t:
#         if s.isupper():
#             res.append(s)
#     return res
# t=['A','b','c']
# print(only_upper(t))
#  train 10.3 -> skip

#### section 10.8
# t=['a','b','c']
# x=t.pop(1)
# print(t)
# print(x)
# t=['a','b','c']
# x=t.pop()
# print(t)
# print(x)
# t=['a','b','c']
# del t[1]
# print(t)
# t=['a','b','c']
# t.remove('b')
# print(t)
# t=['a','b','c','d','e','f']
# del t[1:5]
# print(t)

#### train 10.4
# def middle(t):
#     return t[1:-1]
# print(middle([1,2,3,4]))

#### train 10.5 -> skip
# def chop(t):
#     del t[0]
#     del t[-1]
# t=[1,2,3,4]
# print(t)
# chop(t)
# print(t)

#### section 10.9
# a='spam'
# t=list(a)
# print(t)
# 
# s='pining for the fjords'
# t=s.split()
# print(t)
# 
# s='spam-spam-spam'
# delimiter='-'
# t=s.split(delimiter)
# print(t)
# 
# s='pining for the fjords'
# t=s.split()
# print(t)
# delimiter=' '
# u=delimiter.join(t)
# print(u)

#### section 10.10
# a='banana'
# b='banana'
# print(a is b)
# a=[1,2,3]
# b=[1,2,3]
# print(a is b)

#### section 10.11
# a=[1,2,3]
# b=a
# print(a is b)
# b[0]=17
# print(a)

#### section 10.12
# def delete_head(t):
#     del t[0]
# 
# s=['a','b','c']
# delete_head(s)
# print(s)
# 
# t1=[1,2]
# t2=t1.append(3)
# print(t1)
# print(t2)
# t4=t1+[4]
# print(t4)
# def bad_delete_head(t):
#     t=t[1:]
# print(t4)
# bad_delete_head(t4)
# print(t4)
# def tail(t):
#     return t[1:]
# print(tail(t4))
# print(t4)

#### section 10.13
#### nothing 

#### section 10.14
#### nothing 

#### section 10.13 (typo 10.15?) 
#### train 10.6
# def is_sorted(t):
#     org=t[:]
#     t.sort()
#     return org==t
# print(is_sorted([1,2,3,4]))
# print(is_sorted([1,2,3,0]))

#### train 10.7
# def is_anagram(w1,w2):
#     return sorted(w1)==sorted(w2)
# print(is_anagram('temp','tepm'))

#### train 10.8
# def has_duplicated(s):
#     t=list(s)
#     t.sort()
#     for i in range(len(t)-1):
#         if t[i]==t[i+1]:
#              return True
#     return False
# t=[1,2,3,4]
# print(has_duplicated(t))
# t=[1,2,3,4,4,4,4]
# print(has_duplicated(t))

#### train 10.9
def remove_duplicated(t):
    t.sort()
    for i in range(len(t)-1,1,-1):
        if t[i]==t[i-1]:
             del t[i]
t=[1,2,3,4,4,4,4]
print(t)
remove_duplicated(t)
print(t)

#### train 10.10 -> skip
import time
def make_word_list1():
    t=[]
    fin=open('words.txt')
    for line in fin:
        word=line.strip()
        t.append(word)
    return t
def make_word_list2():
    t=[]
    fin=open('words.txt')
    for line in fin:
        word=line.strip()
        t=t+[word]
    return t

start_time= time.time()
t=make_word_list1()
elapsed_time=time.time()-start_time
print(len(t))
print(t[:10])
print(elapsed_time,'seconds')

start_time= time.time()
t=make_word_list2()
elapsed_time=time.time()-start_time
print(len(t))
print(t[:10])
print(elapsed_time,'seconds')

#### train 10.11 -> skip
#### train 10.12 -> skip
#### train 10.13 -> skip

#### train 10.11 -> skip
#### train 10.12 -> skip
#### train 10.13 -> skip

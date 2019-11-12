#!/usr/bin/env python
# -*- coding: utf-8 -*-

# test06-1.py 
# f = open('test3.txt','r')
# 
# while True:
#     s = f.readline()
#     if s:
#         print(s)
#     else:
#         break
# f.close()

# test06-2.py 
# f = open('test3.txt','r')
# 
# while True:
#     s = f.readline().rstrip()
#     if s:
#         print(s)
#     else:
#         break
# f.close()

# test06-3.py 
f = open('test3.txt','rb')

while True:
    s = f.readline().rstrip().decode('utf-8')
    if s:
        print(s)
    else:
        break
f.close()

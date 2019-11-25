#!/usr/bin/env python
# -*- coding: utf-8 -*-

#### section 14.1
# nothing

#### section 14.2
# fout=open('output.txt','w')
# print(fout)
# line1="This here's the wattle.\m"
# fout.write(line1)
# line2="the emblem of our land. \n"
# fout.write(line2)

#### section 14.3
# x=32
# fout.write(str(x))
# 
# camels=42
# fout.write('%d\n' % camels)
# fout.write('In %d years i have spotted %g %s.' % (3, 0.1, 'camels'))

#### section 14.4
# import os
# cwd = os.getcwd()
# print(cwd)
# print(os.path.abspath('output.txt'))
# print(os.path.exists('output.txt'))
# print(os.path.isdir('output.txt'))
# print(os.path.isfile('output.txt'))
# print(os.listdir(cwd))
# print(os.path.join(cwd,'output.txt'))
# 
# def walk(diename):
#     for name in os.listdir(dirname):
#         path = os.path.join(dirname,name)
#         if os.path.isfile(path):
#             print(path)
#         else:
#             walk(path)

#### train 14.1
# print(os.walk('/home/nakanishi/archive'))
# for dirpath,dirnames,filenames in os.walk('/home/nakanishi/python_work/2018/0410/'):
#     print(dirpath)
#     print(dirnames)
#     print(filenames)
# print(__file__)
# print(os.path.abspath(__file__))
# print(os.path.dirname(os.path.abspath(__file__)))

#### section 14.5
# try:
#     fin=open('bad_file')
#     for line in fin:
#         print(line)
#     fin.close
# except:
#     print('somthing went wrong.')

#### train 14.2 -> skip

#### section 14.6
# import anydbm
# db=anydbm.open('captions.db','c')
# db['cleese.png']='Photo of John Cleese.'
# print(db['cleese.png'])

#### section 14.7
# import pickle
# t1=[1,2,3]
# s=pickle.dumps(t1)
# t2=pickle.loads(s)
# print(pickle.dumps(t1))
# print(t2)
# print(t1==t2)
# print(t1 is t2)

#### train 14.3 -> skip
#### section 14.8
# skip

#### train 14.4 -> skip

#### section 14.9
import wc
print(wc.linecount('wc.py'))

#### train 14.5 -> skip

#### section 14.10
# nothing 
#### section 14.11
# nothing

#### section 14.12
#### train 14.5(typo 14.6?) -> skip

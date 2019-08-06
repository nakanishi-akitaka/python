# -*- coding: utf-8 -*-
"""

pythonで格子点を生成する
https://qiita.com/hitoshi_xyz/items/ec82e108c3ec827512a3

【Python】ふたつの配列からすべての組み合わせを評価
http://kaisk.hatenadiary.com/entry/2014/11/05/041011

[Perl][Python]格子点の生成
http://d.hatena.ne.jp/odz/20070131/1170284561

【numpy】meshgrid メモ
http://umashika5555.hatenablog.com/entry/2017/10/26/043348

配列の要素から格子列を生成するnumpy.meshgrid関数の使い方
https://deepage.net/features/numpy-meshgrid.html

Created on Thu Jan  3 15:24:06 2019

@author: Akitaka
"""
# 二次元の格子点を生成する
import numpy as np
x_min = -5
x_max =  5
n_grid = 1
x_grid = np.linspace(x_min, x_max, n_grid+1)
y_grid = np.linspace(x_min, x_max, n_grid+1)
print(x_grid)
xx, yy = np.meshgrid(x_grid, y_grid)
xy_grid = np.c_[xx.ravel(), yy.ravel()]
print(xy_grid)
print(xy_grid.shape[0])

z = np.array([[0,0]])
z = np.array(np.atleast_2d([0,0]))
print(z)
print(np.r_[xy_grid,z])

def blackbox_func(x):
    """
    Sphere function
    https://qiita.com/tomitomi3/items/d4318bf7afbc1c835dda
    もともとは最小化問題なのでマイナスをかける
    """
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]-1):
            y[i] += 100*(x[i,j+1]-x[i,j]**2)+(x[i,j]-1)**2
    return -y 

print(blackbox_func(xy_grid))
print(blackbox_func(z))

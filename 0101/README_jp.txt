# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 23:20:02 2019

@author: Akitaka
"""

[1b] ベイズ最適化 + kNN + σDD (データ密度によるσ)
ref:20181228
[?] k = 1 とすれば、GPRの時のように、「データがある所では0」とできるのでは？
    あるいは、全てのサンプルからの距離の幾何平均、相乗平均とすれば、データがある所で0 & 連続値
->
[1b1] k = 1 の距離のみ
データのある所では0になる & 95%信頼区間がひし形になる

[1b2] k = all の距離の幾何平均相乗平均(gmean)
https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/
    scipy.stats.mstats.gmean.html
データのある所では0になる & 95%信頼区間がほぼ長方形で変化に乏しい

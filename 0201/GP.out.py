# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 13:05:34 2019

@author: Akitaka
"""

runfile('C:/Users/Akitaka/Downloads/python/0201/GP.py', wdir='C:/Users/Akitaka/Downloads/python/0201')
C:\Users\Akitaka\Anaconda3\lib\site-packages\deap\tools\_hypervolume\pyhv.py:33: ImportWarning: Falling back to the python version of hypervolume module. Expect this to be very slow.
  "module. Expect this to be very slow.", ImportWarning)
C:\Users\Akitaka\Anaconda3\lib\importlib\_bootstrap.py:219: ImportWarning: can't resolve package from __spec__ or __package__, falling back on __name__ and __path__
  return f(*args, **kwds)
C:\Users\Akitaka\Anaconda3\lib\importlib\_bootstrap.py:219: ImportWarning: can't resolve package from __spec__ or __package__, falling back on __name__ and __path__
  return f(*args, **kwds)
                                      fitness                                         size             
                ---------------------------------------------------     -------------------------------
gen     nevals  avg             max     min             std             avg     max     min     std    
0       300     0.741261        0.75243 0.739154        0.000969994     4.52667 14      2       2.84182
1       164     0.742463        0.759495        0.738584        0.00261916      4.42333 14      1       2.70385
2       172     0.745261        0.768907        0.740169        0.0047919       4.73    14      1       3.05458
3       171     0.749827        0.782676        0.740549        0.00735729      5.78333 17      1       3.96271
4       156     0.75783         0.782676        0.740453        0.0105806       10.7967 18      1       4.05241
5       168     0.764185        0.792389        0.740825        0.0141751       13.09   22      3       2.97689
6       167     0.772111        0.793657        0.740836        0.016202        15.1167 22      6       2.57612
7       171     0.775656        0.799636        0.740179        0.0174457       14.83   24      4       2.81681
8       157     0.782075        0.801036        0.737871        0.0180455       15.55   22      1       2.69459
9       166     0.782198        0.801207        0.738945        0.021123        16.2    24      4       2.51131
10      164     0.786271        0.802428        0.740865        0.0212794       16.0267 21      1       2.47372
11 0.8024283025760427 0.8037956478661223 0.7970101383817028 266.8403284549713
                                         fitness                                              size             
                ----------------------------------------------------------      -------------------------------
gen     nevals  avg             max             min             std             avg     max     min     std    
0       300     0.802156        0.814404        0.774752        0.00333762      4.28333 14      2       2.69377
1       164     0.803642        0.814404        0.749265        0.00387871      4.17667 18      1       2.61256
2       164     0.806679        0.81613         0.802141        0.0037591       4.33    10      1       1.86577
3       161     0.80984         0.81613         0.801938        0.00472128      4.64333 11      1       1.68012
4       153     0.81146         0.81613         0.802161        0.00515749      5.13667 11      3       1.38491
5       152     0.81188         0.81613         0.802048        0.00516247      5.62333 9       1       1.19225
6       172     0.811214        0.81613         0.801716        0.00557243      5.66    12      1       1.32831
7       157     0.811631        0.81613         0.770399        0.00579123      5.59333 11      1       1.24952
8       174     0.811479        0.81613         0.802029        0.00534517      5.46333 9       1       1.19805
9       163     0.811848        0.81613         0.802096        0.00520223      5.47    13      1       1.23657
10      147     0.81157         0.81613         0.80202         0.00550522      5.51333 13      1       1.31776
12 0.8161298683240783 0.8172987808734918 0.8114929688000513 317.4050920009613

### Results
Baseline AUC train : 0.7411959004855435
Baseline AUC test : 0.7311947434172593
Best AUC train : 0.8161298683240783
Best AUC test : 0.8114929688000513

### Generated feature expression
mul(sub(add(mul(ARG9, ARG2), cos(neg(ARG6))), neg(ARG6)), sub(ARG7, cos(cos(neg(ARG6)))))
mul(neg(ARG10), mul(ARG9, ARG8))
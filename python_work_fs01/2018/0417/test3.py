#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

y[::5] += 3 * (0.5 - np.random.rand(8))
svr_rbf = SVR(kernel='rbf',    C=1e3, gamma=0.1) 
svr_lin = SVR(kernel='linear', C=1e3)
svr_pol = SVR(kernel='poly',   C=1e3, degree=2) 
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_pol = svr_pol.fit(X, y).predict(X)

lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='r', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='g', lw=lw, label='Linear model')
plt.plot(X, y_pol, color='b', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regresion')
plt.legend()
# plt.show()

# my original

svr_rbf = SVR(kernel='rbf')
svr_lin = SVR(kernel='linear')
svr_pol = SVR(kernel='poly')
svr_sig = SVR(kernel='sigmoid')
print(svr_rbf)
print(svr_lin)
print(svr_pol)
print(svr_sig)

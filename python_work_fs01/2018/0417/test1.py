#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
Cov = np.array([[2.9, -2.2],[-2.2,6.5]])
X = np.random.multivariate_normal([1,2], Cov, size=200)
np.set_printoptions(4, suppress=True)
print(X[:10,])

import matplotlib.pyplot as plt
# plt.figure(figsize=(4,4))
# plt.scatter(X[:,0],X[:,1])
# plt.axis('equal')
# plt.show()

from sklearn.decomposition import PCA
pca = PCA()
X_pca = pca.fit_transform(X)
print(pca.components_)
print(pca.mean_)
print(pca.explained_variance_ratio_)

# plt.figure(figsize=(4,4))
# plt.scatter(X_pca[:,0],X_pca[:,1])
# plt.axis('equal')
# plt.show()

Y = np.dot((X - pca.mean_), pca.components_.T)
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.scatter(X[:,0],X[:,1])
plt.axis('equal')
plt.subplot(122)
plt.scatter(X_pca[:,0],X_pca[:,1])
plt.axis('equal')
# plt.show()

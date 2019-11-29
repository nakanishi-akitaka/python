#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 4.5. Unsupervised dimensionality reduction
# 4.5.1. The Johnson-Lindenstrauss lemma
# 4.5.2. Gaussian random projection
import numpy as np
from sklearn import random_projection
X = np.random.rand(100,10000)
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.shape)

# 4.5.3. Sparse random projection

X = np.random.rand(100,10000)
transformer = random_projection.SparseRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.shape)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 4.3. Preprocessing data
# 4.3.1. Standardization, or mean removal and variance scaling

from sklearn import preprocessing
import numpy as np
X = np.array([
[1., -1., 2.],
[2.,  0., 0.],
[0.,  1.,-1.]
])
X_scaled = preprocessing.scale(X)
print(X_scaled)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

# 4.3.1.1. Scaling features to a range
X_train = np.array([
[1., -1., 2.],
[2.,  0., 0.],
[0.,  1.,-1.]
])
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
print(X_train_minmax)

X_test = np.array([[-3., -1., 4.]])
X_test_minmax = min_max_scaler.transform(X_test)
print(X_test_minmax)
print(min_max_scaler.scale_)
print(min_max_scaler.min_)

X_train = np.array([
[1., -1., 2.],
[2.,  0., 0.],
[0.,  1.,-1.]
])
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
print(X_train_maxabs)

X_test = np.array([[-3., -1., 4.]])
X_test_maxabs = max_abs_scaler.transform(X_test)
print(X_test_maxabs)
print(max_abs_scaler.scale_)

# 4.3.1.2. Scaling sparse data
# nothing

# 4.3.1.3. Scaling data with outliers
# nothing

# 4.3.1.4. Contering kernel matrices
# nothing

# 4.3.2. Normalization
X = [
[1., -1., 2.],
[2.,  0., 0.],
[0.,  1.,-1.]
]
X_normalized = preprocessing.normalize(X, norm='l1')
print('l1', X_normalized)
X_normalized = preprocessing.normalize(X, norm='l2')
print('l2', X_normalized)

normalizer = preprocessing.Normalizer().fit(X)
print(normalizer)
print(normalizer.transform(X))
print(normalizer.transform([[-1.,-1.,0.]]))

# 4.3.3. Binarization
# 4.3.3.1. Feature binarization
X = [ 
[1., -1., 2.],
[2.,  0., 0.],
[0.,  1.,-1.]
]
binarizer = preprocessing.Binarizer().fit(X)
print(binarizer)
print(binarizer.transform(X))
binarizer = preprocessing.Binarizer(threshold=1.1)
print(binarizer.transform(X))

# 4.3.4. Encoding categorical features
enc = preprocessing.OneHotEncoder()
x=enc.fit([[0,0,3],[1,1,0],[0,2,1],[1,0,2]])
print(x)
x=enc.transform([[0,1,3]]).toarray()
print(x)

enc = preprocessing.OneHotEncoder(n_values=[2,3,4])
x=enc.fit([[1,2,3],[0,2,0]])
print(x)
x=enc.transform([[1,0,0]]).toarray()
print(x)

# 4.3.5. Imputation of missing values
import numpy as np
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean',axis=0)
imp.fit([[1,2],[np.nan,3],[7,6]])
X = [[np.nan,2],[6,np.nan],[7,6]]
print(imp.transform(X)) 

import scipy.sparse as sp
X = sp.csc_matrix([[1,2],[0,3],[7,6]])
imp = Imputer(missing_values=0, strategy='mean',axis=0)
imp.fit(X)
X_test = sp.csc_matrix([[0,2],[6,0],[7,6]])
print(imp.transform(X_test)) 

# 4.3.6. Generating polynomial features
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3,2)
print(X)
poly = PolynomialFeatures(2)
print(poly.fit_transform(X))

X = np.arange(9).reshape(3,3)
print(X)
poly = PolynomialFeatures(degree=3,interaction_only=True)
print(poly.fit_transform(X))

# 4.3.7. Custom transformers
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p)
X = np.array([[0,1],[2,3]])
print(transformer.transform(X))



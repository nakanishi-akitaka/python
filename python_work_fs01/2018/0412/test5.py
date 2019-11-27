#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets import load_digits
#  10 = number of class (max 10)
# if load_digits(3), only data 0, 1, 2 
digits = load_digits(10)

# data 
print(digits.data)

# data shape
# 1 data contain 8x8=64 feature 
# number of data = 1797 
print(digits.data.shape)
X_train = digits.data[:1500]
y_train = digits.target[:1500]
X_test = digits.data[1500:]
y_test = digits.target[1500:]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
# learn
lr.fit(X_train, y_train)
# predict
pred = lr.predict(X_test)
print(pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred, labels=digits.target_names)
print(cm)

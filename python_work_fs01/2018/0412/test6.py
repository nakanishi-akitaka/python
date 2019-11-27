#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib

digits = load_digits(10)
X_train = digits.data[:1500]
y_train = digits.target[:1500]
X_test = digits.data[1500:]
y_test = digits.target[1500:]

# make Pipeline
# normalize data & difine SVM
pipeline = Pipeline([
('standard_scaler', StandardScaler()),
('svm', SVC())
])

# determine search range of parameters
# Grid Search YOU no parameters HA MOTTO KMAKAI HOU GA II
params = {
'svm__C' : np.logspace(0, 2, 5),
'svm__gamma' : np.logspace(-3, 0, 5)
}

# Grid Search
clf = GridSearchCV(pipeline, params)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

# reporting of results
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

# save model
# API etc de RIYOU SURU SAI ha joblib.load de HOZON SHITA model wo YOMIKOMU
joblib.dump(clf, 'clf.pkl')

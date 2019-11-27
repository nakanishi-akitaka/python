#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# class: 0
df_a = pd.DataFrame({'x1':np.random.randn(100),
                     'x2':np.random.randn(100),
                     'y' :0})
# class: 1
df_b = pd.DataFrame({'x1':np.random.randn(100)+5,
                     'x2':np.random.randn(100)+3,
                     'y' :1})
df = df_a.append(df_b)
X_train, X_test, y_train, y_test = \
train_test_split(df[['x1','x2']], df['y'], test_size=0.2)

# 1. model instance
clf = SVC()

# 2. fit learning
clf.fit(X_train, y_train)

# 3. predict
y_pred = clf.predict(X_test)

#
# parameter optimization
#
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# search range
param_grid = {
'n_estimators':[2,5,10],
'max_depth':[2,5]
}
rfr=RandomForestClassifier(random_state=0)
grid_search = GridSearchCV(rfr,param_grid,cv=5)

# fit learning with optimized parameters
grid_search.fit(X_train, y_train)
print('test_score : {}'.format(grid_search.score(X_test,y_test)))
print('best_param : {}'.format(grid_search.best_params_))


#
# pipeline
#
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# make pipeline
pipe = Pipeline([
('scaler', StandardScaler()),
('pca', PCA()),
('rf', RandomForestClassifier(random_state=0))
])

# gridsearch hyper parameter
param_grid = {
'pca__n_components' : [1, 2],
'rf__n_estimators' : [2, 10, 100],
'rf__max_depth' : [10, 100, 1000]
}
grid_search = GridSearchCV(pipe,param_grid,cv=5)
grid_search.fit(X_train, y_train)
print('test_score : {}'.format(grid_search.score(X_test,y_test)))
print('best_param : {}'.format(grid_search.best_params_))

# my custom (ref:test6.py)
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
y_pred = grid_search.predict(X_test)
print("pred values = ",list(y_pred))
print("true values = ",list(y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))

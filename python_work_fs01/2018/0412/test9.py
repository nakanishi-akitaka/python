#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 4.1
# 4.1.1 
# 4.1.1.1
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators=[('reduce_dim',PCA()),('clf',SVC())]
pipe = Pipeline(estimators)
# print('pipe = ', pipe)
# print('pipe.steps[0] = ', pipe.steps[0])
# print('pipe.steps[1] = ', pipe.steps[1])
# print('pipe.named_steps[reduce_dim] = ', pipe.named_steps['reduce_dim'])
# print('pipe.named_steps[clf]        = ', pipe.named_steps['clf'])

# print('pipe.steps[1] = ', pipe.steps[1])
# pipe.set_params(clf__C=10)
# print('pipe.steps[1] = ', pipe.steps[1])

from sklearn.model_selection import GridSearchCV
# params = {
# 'reduce_dim__n_components':[2,5,10],
# 'clf__C':[0.1, 10, 100]
# }
params = dict( reduce_dim__n_components=[2,5,10], clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe,param_grid=params)

from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer
mpl=make_pipeline(Binarizer(),MultinomialNB())
# print('make_pipeline = ', mpl)
# print('make_pipeline.steps[0] = ', mpl.steps[0])
# print('make_pipeline.steps[1] = ', mpl.steps[1])

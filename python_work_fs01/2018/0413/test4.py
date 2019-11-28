#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from pprint import pprint
from time import time
import logging
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,format='$(asctime)s (levelname)s %(message)s')

categories = [
'alt.atheism',
'talk.religion.misc'
]
print("Loading 20 news groups dataset for categories:")
print(categories)
data = fetch_20newsgroups(subset='train',categories=categories)
print("%d documents" % len(data.filenames))
print("%d categories" % len(data.target_names))
print()

pipeline = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf',SGDClassifier()),
])

parameters = {
'vect__max_df':(0.5, 0.75, 1.0),
'vect__ngram_range':((1,1),(1,2)),
'clf__alpha':(0.00001, 0.000001),
'clf__penalty':('12', 'elasticnet')
}

if __name__== "__main__":
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    t0 = time()
    grid_search.fit(data.data, data.target)
    print("done in %0.3fs" % (time()-t0))
    print()
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
import numpy as np
import pandas as pd

# scikit-learn 0.18 User Guide 4.2
# 4.2.1. Loading features from dicts
from sklearn.feature_extraction import DictVectorizer
measurements = [
{'city': 'Dubai',         'temperature': 33},
{'city': 'London',        'temperature': 12},
{'city': 'San Fransisco', 'temperature': 18}
]
vec = DictVectorizer()
x=vec.fit_transform(measurements).toarray()
print(x)
x=vec.get_feature_names()
print(x)

pos_window = [{
'word-2': 'the',
'pos-2':  'DT',
'word-1': 'cat',
'pos-2':  'NN',
'word+1': 'on',
'pos+1':  'PP',
}]
vec = DictVectorizer()
x=vec.fit_transform(pos_window)
print(x)
print(x.toarray())
print(vec.get_feature_names())

# 4.2.2. Feature hashing
# skip?

# 4.2.2.1. Implementation details
# nothing

# 4.2.3. Text feature extraction
# 4.2.3.1. The Bag of Words representation
# nothing

# 4.2.3.2. Sparsity
# nothing

# 4.2.3.3. Common Vectorizer usage
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
print(vectorizer)
corpus = [
'This is the first document.',
'This is the second second document.',
'And the third one.',
'Is this the first document?'
]
x = vectorizer.fit_transform(corpus)
print(x)
analyze = vectorizer.build_analyzer()
x = analyze("This is a text document to analyze.") == (['this','is','text','document','to','analyze'])
print(x)
x = vectorizer.get_feature_names() == ([ 'and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this' ])
print(x)
x = vectorizer.fit_transform(corpus)
print(x.toarray())
x = vectorizer.vocabulary_.get('document')
print(x)
x = vectorizer.transform(['Something completely new.']).toarray()
print(x)
bigram_vectorizer = CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b',min_df=1)
analyze = bigram_vectorizer.build_analyzer()
x = analyze("Bi-grams are cool!") == ([ 'bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool' ])
print(x)
x_2 = bigram_vectorizer.fit_transform(corpus).toarray()
print(x_2)
feature_index = bigram_vectorizer.vocabulary_.get('is this')
print(x_2[:,feature_index])

# 4.2.3.4. Tf-idf term weighting
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
print(transformer)
counts=[
[3,0,1],
[2,0,0],
[3,0,0],
[4,0,0],
[3,2,0],
[3,0,2],
]
tfidf = transformer.fit_transform(counts)
print(tfidf)
print(tfidf.toarray())
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(counts).toarray()
print(tfidf)
print(transformer.idf_)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1)
print(vectorizer.fit_transform(corpus))

# 4.2.3.5 Decoding text files
# nothing

# 4.2.3.6. Applications and examples
# nothing

# 4.2.3.7. Limitations of the Bag of Words representation
# skip

# 4.2.3.8. Vectorizing a large text corpus with the hashing trick
# skip

# 4.2.3.9. Performing out-of-core scaling with HashingVectorizer
# skip

# 4.2.3.10. Customizing the vectorizer classes
# skip

# 4.2.4. Image feature extraction
# 4.2.4.1. Patch extraction
from sklearn.feature_extraction import image
one_image = np.arange(4*4*3).reshape((4,4,3))
print(one_image[:,:,0])
patches = image.extract_patches_2d(one_image,(2,2),max_patches=2,random_state=0)
print(patches.shape)
print(patches[:,:,:,0])
patches = image.extract_patches_2d(one_image,(2,2))
print(patches.shape)
print(patches[4,:,:,0])
reconstructed = image.reconstruct_from_patches_2d(patches, (4, 4,3))
np.testing.assert_array_equal(one_image, reconstructed)
five_images = np.arange(5*4*4*3).reshape(5,4,4,3)
patches = image.PatchExtractor((2,2)).transform(five_images)
print(patches.shape)



# 4.2.4.2. Connectivity graph of an image
# nothing


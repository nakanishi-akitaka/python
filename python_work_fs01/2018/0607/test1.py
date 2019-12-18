#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:set foldmethod=marker:
# command table 
# zo: open  1 step (o->O, all step)
# zc: close 1 step (c->C, all step)
# zr: open  all fold 1 step (r->R, all step)
# zm: close all fold 1 step (m->M, all step)
# PEP8
################################################################################
# 80 characters / 1 line
################################################################################

# modules
# {{{
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
# }}}

n_clusters = 3
raw_data = pd.read_csv('dataset/iris.csv',index_col=0)
raw_data_with_y = pd.read_csv('dataset/iris_withspecies.csv',index_col=0)
y = raw_data_with_y[raw_data_with_y.columns[0]]

scaled_data = (raw_data - raw_data.mean(axis=0)) / raw_data.std(axis=0, ddof=1)
clustering = linkage(scaled_data, metric='euclidean', method='ward')
cluster_n = fcluster(clustering, n_clusters,criterion='maxclust')
dendrogram(clustering, labels=raw_data.index, color_threshold=0, orientation='left')
plt.show()

pca = PCA()
scoreT = pca.fit_transform(scaled_data)
plt.scatter(scoreT[:, 0], scoreT[:, 1], c=cluster_n, cmap=plt.get_cmap('jet'))
for i in np.arange(0, raw_data.shape[0] - 1):
    plt.text(scoreT[i, 0], scoreT[i, 1], raw_data.index[i],
    horizontalalignment='left', verticalalignment='top')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()

plt.scatter(scoreT[:, 0], scoreT[:, 1], c=cluster_n, cmap=plt.get_cmap('jet'))
for i in np.arange(0, raw_data.shape[0] - 1):
    plt.text(scoreT[i, 0], scoreT[i, 1], y[i+1],
    horizontalalignment='left', verticalalignment='top')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()

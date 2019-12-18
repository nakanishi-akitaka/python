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
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
# }}}

name = 'dataset/iris.csv'
raw_data = pd.read_csv(name,index_col=0)
scaled_data = (raw_data - raw_data.mean(axis=0)) / raw_data.std(axis=0, ddof=1)

pca = PCA()
scoreT = pca.fit_transform(scaled_data)
loading_vector = pca.components_.transpose()

contribution_ratio = pca.explained_variance_ratio_
cumulative_contribution_ratio = np.cumsum(contribution_ratio)

data1 = contribution_ratio
data2 = cumulative_contribution_ratio
plt.bar (np.arange(1, len(data1) + 1), data1, align='center')
plt.plot(np.arange(1, len(data2) + 1), data2, 'r-')
plt.ylim(0, 1)
plt.xlabel('Number of PCs')
plt.ylabel('Contribution ratio(blue), Cumulative contribution ratio(red)')
plt.show()

plt.scatter(scoreT[:, 0], scoreT[:, 1])
for i in np.arange(0, raw_data.shape[0] - 1):
    plt.text(scoreT[i, 0], scoreT[i, 1], raw_data.index[i],
    horizontalalignment='left', verticalalignment='top')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()

if scoreT.shape[1] > 3:
    ScoreTpd = pd.DataFrame(scoreT[:, :4])
    ScoreTpd.columns = ['PC1', 'PC2', 'PC3', 'PC4']
    scatter_matrix(ScoreTpd)
    plt.show()


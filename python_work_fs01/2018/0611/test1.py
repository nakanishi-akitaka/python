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
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
# }}}

regression_method_flag = 1 # 1:OLS, 2:PLS(constant component number), 3:PLS
pls_component_number = 2
max_pls_component_number = 50
fold_number = 5
threshold_of_rate_of_same_value = 0.79 # threshold of same value
do_autoscaling = True # True or False

raw_data_with_y = pd.read_csv('dataset/logSdataset1290.csv',index_col=0)

raw_data_with_y = raw_data_with_y.loc[:, raw_data_with_y.mean().index]
raw_data_with_y = raw_data_with_y.replace(np.inf, np.nan).fillna(np.nan)
raw_data_with_y = raw_data_with_y.dropna(axis=1)
raw_data_with_y = raw_data_with_y.loc[~raw_data_with_y.index.duplicated(keep=False),:]
y = raw_data_with_y[raw_data_with_y.columns[0]]
rawX = raw_data_with_y[raw_data_with_y.columns[1:]]
rawX_tmp = rawX.copy()

# delete descriptors with high rate of the same values
rate_of_same_value = list()
num = 0
for X_variable_name in rawX.columns:
    num += 1
    print('{0} / {1}'.format(num, rawX.shape[1]))
    same_value_number = rawX[X_variable_name].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / rawX.shape[0]))
deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)

deleting_variable_numbers = np.where(rawX.var() == 0)

if len(deleting_variable_numbers[0]) == 0:
    X = rawX
else:
    X = rawX.drop(rawX.columns[deleting_variable_numbers], axis=1)
    print('Variable numbers zero variance: {0}'.format(deleting_variable_numbers[0] + 1))
print('# of X-variables: {0}'.format(X.shape[1]))

if do_autoscaling:
    autoscaled_X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    autoscaled_y = (y - y.mean()) / y.std(ddof=1)
else:
    autoscaled_X = X
    autoscaled_y = y

if regression_method_flag == 1:
    regression_model = LinearRegression()
elif regression_method_flag == 2:
    regression_model = PLSRegression(n_components=pls_component_number)
elif regression_method_flag == 3:
    pls_components = np.arange(1, min(np.linalg.matrix_rank(autoscaled_X) + 1, max_pls_component_number + 1), 1)
    r2all = list()
    r2cvall = list()
    for pls_component in pls_components:
        pls_model_in_cv = PLSRegression(n_components=pls_component)
        pls_model_in_cv.fit(autoscaled_X, autoscaled_y)
        calculated_y_in_cv = np.ndarray.flatten(pls_model_in_cv.predict(autoscaled_X))
        estimated_y_in_cv = np.ndarray.flatten(
            model_selection.cross_val_predict(pls_model_in_cv, autoscaled_X, autoscaled_y, cv=fold_number))  
        if do_autoscaling:
            calculated_y_in_cv = calculated_y_in_cv * y.std(ddof=1) + y.mean()
            estimated_y_in_cv = estimated_y_in_cv * y.std(ddof=1) + y.mean()
        r2all.append(float(1 - sum((y - calculated_y_in_cv) **2) / sum((y - y.mean()) **2 )))
        r2cvall.append(float(1 - sum((y - estimated_y_in_cv) **2) / sum((y - y.mean()) **2 )))
    plt.plot(pls_components, r2all, 'bo-')
    plt.plot(pls_components, r2cvall, 'ro-')
    plt.ylim(0, 1)
    plt.xlabel('Number of PLS components')
    plt.ylabel('r2(blue), r2cv(red)')
    plt.show()
    optimal_pls_component_number = np.where(r2cvall == np.max(r2cvall))
    optimal_pls_component_number = optimal_pls_component_number[0][0] + 1
    regression_model = PLSRegression(n_components=optimal_pls_component_number)

regression_model.fit(autoscaled_X, autoscaled_y)

calculated_y = np.ndarray.flatten(regression_model.predict(autoscaled_X))
estimated_y = np.ndarray.flatten(
    model_selection.cross_val_predict(regression_model, autoscaled_X, autoscaled_y, cv=fold_number))

if do_autoscaling:
    calculated_y = calculated_y * y.std(ddof=1) + y.mean()
    estimated_y = estimated_y * y.std(ddof=1) + y.mean()

print('r2: {0}'.format(float(1 - sum((y  - calculated_y) **2) / sum((y - y.mean()) ** 2))))

plt.figure(figsize=figure.figaspect(1))
plt.scatter(y, calculated_y)
YMax = np.max(np.array([np.array(y), calculated_y]))
YMin = np.min(np.array([np.array(y), calculated_y]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim( YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim( YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel('Actual Y')
plt.ylabel('Calculated Y')
plt.show()

plt.figure(figsize=figure.figaspect(1))
plt.scatter(y, estimated_y)
YMax = np.max(np.array([np.array(y), estimated_y]))
YMin = np.min(np.array([np.array(y), estimated_y]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim( YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim( YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel('Actual Y')
plt.ylabel('Estimated Y')
plt.show()

standard_regression_coefficients = regression_model.coef_
standard_regression_coefficients = pd.DataFrame(standard_regression_coefficients)
standard_regression_coefficients.index = X.columns
standard_regression_coefficients.columns = ['standard regression coefficient']
standard_regression_coefficients.to_csv('standard_regression_coefficients.csv')

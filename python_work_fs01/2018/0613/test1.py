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

regression_method_flag = 3 # 1:OLS, 2:PLS(constant component number), 3:PLS
pls_component_number = 2
max_pls_component_number = 50
fold_number = 2
threshold_of_rate_of_same_value = 0.79 # threshold of same value
do_autoscaling = True # True or False
number_of_training_data = 878
k_in_kNN = 10

# load data set
raw_data_with_y = pd.read_csv('dataset/logSdataset1290.csv',index_col=0)
raw_data_with_y = raw_data_with_y.drop(['Ipc', 'Kappa3'], axis=1)
# Kappa3: 889 dake hazure atai wo motsu kijutushi

ytrain = raw_data_with_y.iloc[:number_of_training_data , 0]
ytest  = raw_data_with_y.iloc[ number_of_training_data:, 0]
raw_Xtrain = raw_data_with_y.iloc[:number_of_training_data , 1:]
raw_Xtest  = raw_data_with_y.iloc[ number_of_training_data:, 1:]

# delete descriptors with high rate of the same values
rate_of_same_value = list()
num = 0
for X_variable_name in raw_Xtrain.columns:
    num += 1
    print('{0} / {1}'.format(num, raw_Xtrain.shape[1]))
    same_value_number = raw_Xtrain[X_variable_name].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / raw_Xtrain.shape[0]))
deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)

"""
deleting_variable_numbers = np.where(raw_Xtrain.var() == 0)
"""

if len(deleting_variable_numbers[0]) == 0:
    Xtrain = raw_Xtrain.copy()
    Xtest  = raw_Xtest.copy()
else:
    Xtrain = raw_Xtrain.drop(raw_Xtrain.columns[deleting_variable_numbers], axis=1)
    Xtest  = raw_Xtest.drop(raw_Xtest.columns[deleting_variable_numbers], axis=1)
    print('Variable numbers zero variance: {0}'.format(deleting_variable_numbers[0] + 1))
print('# of X-variables: {0}'.format(Xtrain.shape[1]))

if do_autoscaling:
    autoscaled_Xtrain = (Xtrain - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)
    autoscaled_Xtest  = (Xtest  - Xtest.mean(axis=0)) / Xtest.std(axis=0, ddof=1)
    autoscaled_ytrain = (ytrain - ytrain.mean()) / ytrain.std(ddof=1)
else:
    autoscaled_Xtrain = Xtrain.copy()
    autoscaled_Xtest  = Xtest.copy()
    autoscaled_ytrain = ytrain.copy()

if regression_method_flag == 1:
    regression_model = LinearRegression()
elif regression_method_flag == 2:
    regression_model = PLSRegression(n_components=pls_component_number)
elif regression_method_flag == 3:
    pls_components = np.arange(1, min(np.linalg.matrix_rank(autoscaled_Xtrain) + 1, max_pls_component_number + 1), 1)
    r2all = list()
    r2cvall = list()
    for pls_component in pls_components:
        pls_model_in_cv = PLSRegression(n_components=pls_component)
        pls_model_in_cv.fit(autoscaled_Xtrain, autoscaled_ytrain)
        calculated_y_in_cv = np.ndarray.flatten(pls_model_in_cv.predict(autoscaled_Xtrain))
        estimated_y_in_cv = np.ndarray.flatten(
            model_selection.cross_val_predict(pls_model_in_cv, autoscaled_Xtrain, autoscaled_ytrain, cv=fold_number))  
        if do_autoscaling:
            calculated_y_in_cv = calculated_y_in_cv * ytrain.std(ddof=1) + ytrain.mean()
            estimated_y_in_cv = estimated_y_in_cv * ytrain.std(ddof=1) + ytrain.mean()
        r2all.append(float(1 - sum((ytrain - calculated_y_in_cv) **2) / sum((ytrain - ytrain.mean()) **2 )))
        r2cvall.append(float(1 - sum((ytrain - estimated_y_in_cv) **2) / sum((ytrain - ytrain.mean()) **2 )))
    plt.plot(pls_components, r2all, 'bo-')
    plt.plot(pls_components, r2cvall, 'ro-')
    plt.ylim(0, 1)
    plt.xlabel('Number of PLS components')
    plt.ylabel('r2(blue), r2cv(red)')
    plt.show()
    optimal_pls_component_number = np.where(r2cvall == np.max(r2cvall))
    optimal_pls_component_number = optimal_pls_component_number[0][0] + 1
    regression_model = PLSRegression(n_components=optimal_pls_component_number)

regression_model.fit(autoscaled_Xtrain, autoscaled_ytrain)

# calcutate y
calculated_ytrain = np.ndarray.flatten(regression_model.predict(autoscaled_Xtrain))
if do_autoscaling:
    calculated_ytrain = calculated_ytrain * ytrain.std(ddof=1) + ytrain.mean()

# set AD
knn_distance = list()
autoscaled_Xtrain_np = np.array(autoscaled_Xtrain)
for training_sample_number_1 in range(autoscaled_Xtrain.shape[0]):
    distance_all = list()
    for training_sample_number_2 in range(autoscaled_Xtrain.shape[0]):
        distance_all.append(np.linalg.norm(
            autoscaled_Xtrain_np[training_sample_number_1, :] - autoscaled_Xtrain_np[training_sample_number_2, :]))
    distance_all.sort()
    knn_distance.append(np.mean(distance_all[1:k_in_kNN + 1]))
# set threshold
knn_distance.sort()
ad_threshold = knn_distance[round(autoscaled_Xtrain.shape[0] * 0.997) - 1]

# r2
print('r2:   {0}'.format(float(1 - sum((ytrain  - calculated_ytrain) **2) / sum((ytrain - ytrain.mean()) ** 2))))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(ytrain, calculated_ytrain)
YMax = np.max(np.array([np.array(ytrain), calculated_ytrain]))
YMin = np.min(np.array([np.array(ytrain), calculated_ytrain]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim( YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim( YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel('Actual Y')
plt.ylabel('Calculated Y')
plt.show()

# standard regression coefficients
standard_regression_coefficients = regression_model.coef_
standard_regression_coefficients = pd.DataFrame(standard_regression_coefficients)
standard_regression_coefficients.index = Xtrain.columns
standard_regression_coefficients.columns = ['standard regression coefficient']
standard_regression_coefficients.to_csv('standard_regression_coefficients.csv')

# prediction
predicted_ytest = np.ndarray.flatten(regression_model.predict(autoscaled_Xtest))
if do_autoscaling:
    predicted_ytest = predicted_ytest * ytrain.std(ddof=1) + ytrain.mean()

# AD check
ad_inside_flag = list()
knn_distance_test = list()
autoscaled_Xtest_np = np.array(autoscaled_Xtest)
for test_sample_number in range(autoscaled_Xtest.shape[0]):
    distance_all = list()
    for training_sample_number in range(autoscaled_Xtrain.shape[0]):
        distance_all.append(
            np.linalg.norm(
                autoscaled_Xtest_np[test_sample_number, :] - autoscaled_Xtrain_np[training_sample_number, :]))
    distance_all.sort()
    knn_distance_test.append(np.mean(distance_all[0:k_in_kNN]))
    if np.mean(distance_all[0:k_in_kNN]) < ad_threshold:
        ad_inside_flag.append(1)
    else:
        ad_inside_flag.append(0)

# statistics and yy-plot
ytest_inside_ad = ytest[knn_distance_test <= ad_threshold]
ytest_outside_ad = ytest[knn_distance_test > ad_threshold]
predicted_ytest_inside_ad = predicted_ytest[knn_distance_test <= ad_threshold]
predicted_ytest_outside_ad = predicted_ytest[knn_distance_test > ad_threshold]

if ytest_inside_ad.shape[0] > 0:
    print('r2p inside of AD: {0}'.format(float(
        1 - sum((ytest_inside_ad - predicted_ytest_inside_ad) ** 2) / sum(
            (ytest_inside_ad - ytest_inside_ad.mean()) ** 2))))
    print('RMSEp inside of AD: {0}'.format(
        float((sum((ytest_inside_ad - predicted_ytest_inside_ad) ** 2) / ytest_inside_ad.shape[0]) ** 0.5)))
    print('MAEp inside of AD: {0}'.format(
        float(sum(abs(ytest_inside_ad - predicted_ytest_inside_ad)) / ytest_inside_ad.shape[0])))
    # yy-plot inside AD
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(ytest_inside_ad, predicted_ytest_inside_ad)
    YMax = np.max(np.array([np.array(ytest_inside_ad), predicted_ytest_inside_ad]))
    YMin = np.min(np.array([np.array(ytest_inside_ad), predicted_ytest_inside_ad]))
    plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
             [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
    plt.ylim( YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
    plt.xlim( YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
    plt.xlabel('Actual Y inside AD')
    plt.ylabel('Predicted Y inside AD')
    plt.show()

if ytest_outside_ad.shape[0] > 0:
    print('r2p outside of AD: {0}'.format(float(1 - sum((ytest_outside_ad - predicted_ytest_outside_ad) ** 2) / sum(
            (ytest_outside_ad - ytest_outside_ad.mean()) ** 2))))
    print('RMSEp outside of AD: {0}'.format(
        float((sum((ytest_outside_ad - predicted_ytest_outside_ad) ** 2) / ytest_outside_ad.shape[0]) ** 0.5)))
    print('MAEp outside of AD: {0}'.format(
        float(sum(abs(ytest_outside_ad - predicted_ytest_outside_ad)) / ytest_outside_ad.shape[0])))
    # yy-plot outside AD
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(ytest_outside_ad, predicted_ytest_outside_ad)
    YMax = np.max(np.array([np.array(ytest_outside_ad), predicted_ytest_outside_ad]))
    YMin = np.min(np.array([np.array(ytest_outside_ad), predicted_ytest_outside_ad]))
    plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
             [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
    plt.ylim( YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
    plt.xlim( YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
    plt.xlabel('Actual Y outside AD')
    plt.ylabel('Predicted Y outside AD')
    plt.show()

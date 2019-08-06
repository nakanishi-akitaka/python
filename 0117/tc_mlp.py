# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:37:25 2019

@author: Akitaka

"""


import numpy as np
import pandas as pd
from time                             import time
from pymatgen                         import Composition
from sklearn.model_selection          import GridSearchCV, KFold
from sklearn.model_selection          import RandomizedSearchCV
from sklearn.preprocessing            import StandardScaler
from sklearn.model_selection          import train_test_split
from sklearn.neighbors                import NearestNeighbors
from sklearn.metrics                  import mean_absolute_error
from sklearn.metrics                  import mean_squared_error
from sklearn.metrics                  import r2_score
from sklearn.model_selection          import cross_val_predict
from sklearn.neural_network           import MLPRegressor

start = time()
#%%
# function

def get_parameters(formula):
    """
    make parameters from chemical formula
    
    Parameters
    ----------
    formula : string
        chemical formula

    Returns
    -------
    array-like, shape = [2*numbers of atom]
        atomic number Z, numbers of atom
    """
    material = Composition(formula)
    features = []
    atomicNo = []
    natom = []
    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
        atomicNo.append(float(element.Z))
    features.extend(atomicNo)
    features.extend(natom)
    return features

def read_fxy_csv(name): 
    """
    read chemical formula, X, y from csv file
    
    Parameters
    ----------
    name : string
        csv file name

    Returns
    -------
    f : array-like, shape = [n_samples]
        chemical formulas
    X : array-like, shape = [n_samples, n_features]
        input parameters
    y : array-like, shape = [n_samples]
        output parameters
        
    """
    data = np.array(pd.read_csv(filepath_or_buffer=name, index_col=0,
                                header=0, sep=','))[:,:]
    f = np.array(data[:,0],dtype=np.unicode)
    y = np.array(data[:,1],dtype=np.float)
    X = np.array(data[:,2:],dtype=np.float)
    return f, X, y


def print_gscv_score_rgr(gscv, X_train, X_test, y_train, y_test, cv):
    """
    print score of results of GridSearchCV (regression)

    Parameters
    ----------
    gscv : 
        GridSearchCV (scikit-learn)

    X_train : array-like, shape = [n_samples, n_features]
        X training data

    y_train : array-like, shape = [n_samples]
        y training data

    X_test : array-like, sparse matrix, shape = [n_samples, n_features]
        X test data

    y_test : array-like, shape = [n_samples]
        y test data

    cv : int, cross-validation generator or an iterable
        ex: 3, 5, KFold(n_splits=5, shuffle=True)

    Returns
    -------
    None
    """
    print()
    print("Best parameters set found on development set:")
    print(gscv.best_params_)
    y_calc = gscv.predict(X_train)
    rmse  = np.sqrt(mean_squared_error (y_train, y_calc))
    mae   =         mean_absolute_error(y_train, y_calc)
    r2    =         r2_score           (y_train, y_calc)
    print('C:  RMSE, MAE, R^2 = {:6.3f}, {:6.3f}, {:6.3f}'\
    .format(rmse, mae, r2))

    y_incv = cross_val_predict(gscv, X_train, y_train, cv=cv)
    rmse  = np.sqrt(mean_squared_error (y_train, y_incv))
    mae   =         mean_absolute_error(y_train, y_incv)
    r2    =         r2_score           (y_train, y_incv)
    print('CV: RMSE, MAE, R^2 = {:6.3f}, {:6.3f}, {:6.3f}'\
    .format(rmse, mae, r2))

    y_pred = gscv.predict(X_test)
    rmse  = np.sqrt(mean_squared_error (y_test, y_pred))
    mae   =         mean_absolute_error(y_test, y_pred)
    r2    =         r2_score           (y_test, y_pred)
    print('TST:RMSE, MAE, R^2 = {:6.3f}, {:6.3f}, {:6.3f}'\
    .format(rmse, mae, r2))
    print()


def ad_knn(X_train, X_test):
    """
    Determination of Applicability Domain (k-Nearest Neighbor)
    
    Parameters
    ----------
    X_train : array-like, shape = [n_samples, n_features]
        X training data

    X_test : array-like, shape = [n_samples, n_features]
        X test data

    Returns
    -------
    array-like, shape = [n_samples]
        -1 (outer of AD) or 1 (inner of AD)
    """
    n_neighbors = 5      # number of neighbors
    r_ad = 0.9           # ratio of X_train inside AD / all X_train
    # ver.1
    neigh = NearestNeighbors(n_neighbors=n_neighbors+1)
    neigh.fit(X_train)
    dist_list = np.mean(neigh.kneighbors(X_train)[0][:,1:], axis=1)
    dist_list.sort()
    ad_thr = dist_list[round(X_train.shape[0] * r_ad) - 1]
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X_train)
    dist = np.mean(neigh.kneighbors(X_test)[0], axis=1)
    y_appd = 2 * (dist < ad_thr) -1

    return y_appd


nprm = len(get_parameters('H3S'))

name = 'mlp'
model = MLPRegressor()
"""
    hidden_layer_sizes only optimization
    "hidden_layer_sizes":[(1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,)],
        'hidden_layer_sizes': (5,),
        C:  RMSE, MAE, R^2 = 21.728, 14.859,  0.860
        CV: RMSE, MAE, R^2 = 37.364, 23.231,  0.585
        TST:RMSE, MAE, R^2 = 25.679, 17.761,  0.790
        17.97 seconds 

    "hidden_layer_sizes":[(10,),(20,),(40,),(60,),(80,),(100,)],
        'hidden_layer_sizes': (100,),
        C:  RMSE, MAE, R^2 = 13.991,  9.657,  0.942
        CV: RMSE, MAE, R^2 = 32.144, 20.683,  0.693
        TST:RMSE, MAE, R^2 = 14.814, 10.766,  0.930
        27.27 seconds 

    "hidden_layer_sizes":[(100,),(200,),(400,),(600,),(800,),(1000,)],
        'hidden_layer_sizes': (100,)
        C:  RMSE, MAE, R^2 = 13.991,  9.657,  0.942
        CV: RMSE, MAE, R^2 = 33.734, 20.198,  0.661
        TST:RMSE, MAE, R^2 = 14.814, 10.766,  0.930
        223.59 seconds 

    "hidden_layer_sizes":[(1,1),(2,2),(4,4),(6,6),(8,8),(10,10)],
        'hidden_layer_sizes': (10, 10)
        C:  RMSE, MAE, R^2 = 17.718, 11.908,  0.907
        CV: RMSE, MAE, R^2 = 26.427, 17.816,  0.792
        TST:RMSE, MAE, R^2 = 22.509, 14.932,  0.839
        14.58 seconds 

    "hidden_layer_sizes":[(10,10),(20,20),(40,40),(60,60),(80,80),(100,100)],
        'hidden_layer_sizes': (100, 100)
        C:  RMSE, MAE, R^2 = 12.220,  8.194,  0.956
        CV: RMSE, MAE, R^2 = 26.836, 17.175,  0.786
        TST:RMSE, MAE, R^2 = 16.390, 11.506,  0.915
        53.60 seconds 

    "hidden_layer_sizes":[(1,1,1),(2,2,2),(4,4,4),(6,6,6),(8,8,8),(10,10,10)],
        'hidden_layer_sizes': (8, 8, 8)
        C:  RMSE, MAE, R^2 = 16.841, 11.897,  0.916
        CV: RMSE, MAE, R^2 = 28.836, 17.969,  0.753
        TST:RMSE, MAE, R^2 = 18.454, 13.236,  0.892
        15.05 seconds 

    "hidden_layer_sizes":[(10,10,10),(20,20,20),(40,40,40),(60,60,60),(80,80,80),(100,100,100)],
        'hidden_layer_sizes': (40, 40, 40)
        C:  RMSE, MAE, R^2 = 12.544,  8.379,  0.953
        CV: RMSE, MAE, R^2 = 29.972, 17.897,  0.733
        TST:RMSE, MAE, R^2 = 12.745,  9.326,  0.948
        81.61 seconds 

    "hidden_layer_sizes":[(1,1,1,1),(2,2,2,2),(4,4,4,4),(6,6,6,6),(8,8,8,8),(10,10,10,10)],
        'hidden_layer_sizes': (10, 10, 10, 10)
        C:  RMSE, MAE, R^2 = 15.705, 10.622,  0.927
        CV: RMSE, MAE, R^2 = 32.173, 21.041,  0.692
        TST:RMSE, MAE, R^2 = 20.760, 13.577,  0.863
        18.93 seconds 

    "hidden_layer_sizes":[(10,10,10,10),(20,20,20,20),(40,40,40,40),(60,60,60,60),(80,80,80,80),(100,100,100,100)],
        'hidden_layer_sizes': (80, 80, 80, 80)
        C:  RMSE, MAE, R^2 = 12.749,  8.838,  0.952
        CV: RMSE, MAE, R^2 = 27.362, 17.170,  0.777
        TST:RMSE, MAE, R^2 = 15.362, 11.006,  0.925
        114.86 seconds 

    "hidden_layer_sizes":[(1,1,1,1,1),(2,2,2,2,2),(4,4,4,4,4),(6,6,6,6,6),(8,8,8,8,8),(10,10,10,10,10)],
        'hidden_layer_sizes': (10, 10, 10, 10, 10)
        C:  RMSE, MAE, R^2 = 18.687, 12.667,  0.896
        CV: RMSE, MAE, R^2 = 31.948, 21.124,  0.696
        TST:RMSE, MAE, R^2 = 23.554, 16.902,  0.824
        16.91 seconds 

    "hidden_layer_sizes":[(10,10,10,10,10),(20,20,20,20,20),(40,40,40,40,40),(60,60,60,60,60),(80,80,80,80,80),(100,100,100,100,100)],
        'hidden_layer_sizes': (80, 80, 80, 80, 80)
        C:  RMSE, MAE, R^2 = 12.797,  8.777,  0.951
        CV: RMSE, MAE, R^2 = 24.854, 15.826,  0.816
        TST:RMSE, MAE, R^2 = 15.864, 11.310,  0.920
        157.85 seconds 

    "hidden_layer_sizes":[(1,1,1,1,1,1),(2,2,2,2,2,2),(4,4,4,4,4,4),(6,6,6,6,6,6),(8,8,8,8,8,8),(10,10,10,10,10,10)],
        'hidden_layer_sizes': (10, 10, 10, 10, 10, 10)
        C:  RMSE, MAE, R^2 = 17.379, 11.722,  0.910
        CV: RMSE, MAE, R^2 = 27.521, 17.424,  0.775
        TST:RMSE, MAE, R^2 = 22.071, 15.045,  0.845
        16.14 seconds 

"""
param_grid={
    "hidden_layer_sizes":[(10,10,10,10,10),(20,20,20,20,20),(40,40,40,40,40),(60,60,60,60,60),(80,80,80,80,80),(100,100,100,100,100)],
        "activation":['relu'], #['identity', 'logistic', 'tanh', 'relu'],
        "solver":['lbfgs'], # ['lbfgs', 'sgd', 'adam'],
        "alpha":[0.0001], # np.logspace(-5, -2, 4),
        "batch_size":['auto'],
        "learning_rate_init":[0.001],
        "learning_rate":['constant'],
        "power_t":[0.5],
        "max_iter":[200],
        "shuffle":[True],
        "random_state":[0],
        "tol":[0.0001], # np.logspace(-5, -3, 3),
        "verbose":[False],
        "warm_start":[False],
        "momentum":[0.9],
        "nesterovs_momentum":[True],
        "early_stopping":[True],
        "validation_fraction":[0.1],
        "beta_1":[0.9],
        "beta_2":[0.999],
        "epsilon":[1e-9],
}

output = 'tc_' + name + '.csv'

print()
print('read train & pred data from csv file')
print()
data_file = 'tc_data.csv'
_, X, y = read_fxy_csv(data_file)
pred_file = 'tc_pred.csv'
f_pred, X_pred, _ = read_fxy_csv(pred_file)

scaler = StandardScaler()

X = scaler.fit_transform(X)
P_pred = X_pred[:,0]
X_pred = scaler.transform(X_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%%
# Set the parameters by cross-validation

n_splits = 3
cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)

n_iter=20
gscv = GridSearchCV(model, param_grid, cv=cv)
#gscv = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=cv)
gscv.fit(X_train, y_train)
print_gscv_score_rgr(gscv, X_train, X_test, y_train, y_test, cv)

if(True):
    #%%
    # Re-learning with all data & best parameters -> Prediction
    best = gscv.best_estimator_.fit(X, y)
    y_pred = best.predict(X_pred)
    
    #%%
    # Applicability Domain (inside: +1, outside: -1)
    
    y_appd = ad_knn(X, X_pred)
    
    data = []
    for i in range(len(X_pred)):
        temp = (f_pred[i], int(P_pred[i]), int(y_pred[i]), y_appd[i])
        data.append(temp)
    
    properties=['formula','P', 'Tc', 'AD']
    df = pd.DataFrame(data, columns=properties)
    df.sort_values('Tc', ascending=False, inplace=True)
    
    df.to_csv(output, index=False)
    print('Predicted Tc is written in file {}'.format(output))

print('{:.2f} seconds '.format(time() - start))

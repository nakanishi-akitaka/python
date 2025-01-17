import numpy as np
#import pandas as pd
from sklearn import model_selection, svm, datasets
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import time
# Settings
svrcs       = 2**np.arange( -5,  10, dtype=float)
svrepsilons = 2**np.arange( -10,  0, dtype=float)
svrgammas   = 2**np.arange( -20, 10, dtype=float)
nfold = 5
ntrain = 1000
ntest  = 1000
# Generate samples demonstration
X, y = datasets.make_regression(n_samples=ntrain+ntest, n_features=100,
                                n_informative=100, noise=100, random_state=0)
# Divide samples into trainign samples and test samples
Xtrain, Xtest, ytrain, ytest \
 = train_test_split(X, y, test_size=ntest, random_state=0)

# Standarize X and y
std_Xtrain = (Xtrain - Xtrain.mean()) / Xtrain.std(ddof=1)
std_ytrain = (ytrain - ytrain.mean()) / ytrain.std(ddof=1)
std_Xtest = (Xtest - Xtrain.mean()) / Xtrain.std(ddof=1)

# Measure time in hyperparameter optimization
starttime = time.time()

# Optimize gamma by maximizing variance in Gram matrix
np_std_Xtrain = np.array(std_Xtrain)
varianceofgrammatrix = list()
for svrgamma in svrgammas:
    grammatrix = np.exp(-svrgamma*((np_std_Xtrain[:, np.newaxis] - np_std_Xtrain)**2).sum(axis=2))
    print(grammatrix)
    varianceofgrammatrix.append(grammatrix.var(ddof=1))
optimalsvrgamma = svrgammas[ np.where( varianceofgrammatrix == np.max(varianceofgrammatrix) )[0][0] ]

# Optimize epsilon with cross-validation
svrmodelincv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimalsvrgamma), {'epsilon':svrepsilons}, cv=nfold )
svrmodelincv.fit(std_Xtrain, std_ytrain)
optimalsvrepsilon = svrmodelincv.best_params_['epsilon']

# Optimize C with cross-validation
svrmodelincv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimalsvrepsilon, gamma=optimalsvrgamma), {'C':svrcs}, cv=nfold )
svrmodelincv.fit(std_Xtrain, std_ytrain)
optimalsvrc = svrmodelincv.best_params_['C']

# Optimize gamma with cross-validation (optional)
svrmodelincv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimalsvrepsilon, C=optimalsvrc), {'gamma':svrgammas}, cv=nfold )
svrmodelincv.fit(std_Xtrain, std_ytrain)
optimalsvrgamma = svrmodelincv.best_params_['gamma']

# Check time in hyperparameter optimization
elapsedtime = time.time() - starttime
print("Elaosed time in hyperparameter optimization: {0} [sec]".format(elapsedtime))
# Check optimized hyperparameters
print ("C: {0}, Epsilon: {1}, Gamma: {2}".format(optimalsvrc, optimalsvrepsilon, optimalsvrgamma))
# Construct SVR model
regressionmodel = svm.SVR(kernel='rbf', C=optimalsvrc, epsilon=optimalsvrepsilon, gamma=optimalsvrgamma)
regressionmodel.fit(std_Xtrain, std_ytrain)
# Calculate y of trainig dataset
calculatedytrain = np.ndarray.flatten( regressionmodel.predict(std_Xtrain) )
calculatedytrain = calculatedytrain*ytrain.std(ddof=1) + ytrain.mean()
# r2, RMSE, MAE
print( "r2: {0}".format(float( 1 - sum( (ytrain-calculatedytrain )**2 ) / sum((ytrain-ytrain.mean())**2) )) )
print( "RMSE: {0}".format(float( ( sum( (ytrain-calculatedytrain)**2 )/ len(ytrain))**0.5 )) )
print( "MAE: {0}".format(float( sum( abs(ytrain-calculatedytrain)) / len(ytrain) )) )
# yyplot
plt.figure(figsize=figure.figaspect(1))
plt.scatter( ytrain, calculatedytrain)
YMax = np.max( np.array([np.array(ytrain), calculatedytrain]))
YMin = np.min( np.array([np.array(ytrain), calculatedytrain]))
plt.plot([YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin)], [YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin)], 'k-')
plt.ylim(YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin))
plt.xlim(YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin))
plt.xlabel("Actual Y")
plt.ylabel("Calculated Y")
plt.show()

# Estimate y in cross-validation
estimatedyincv = np.ndarray.flatten( model_selection.cross_val_predict(regressionmodel, std_Xtrain, std_ytrain, cv=nfold) )
estimatedyincv = estimatedyincv*ytrain.std(ddof=1) + ytrain.mean()
# r2cv, RMSEcv, MAEcv
print( "r2cv: {0}".format(float( 1 - sum( (ytrain-estimatedyincv )**2 ) / sum((ytrain-ytrain.mean())**2) )) )
print( "RMSEcv: {0}".format(float( ( sum( (ytrain-estimatedyincv)**2 )/ len(ytrain))**0.5 )) )
print( "MAEcv: {0}".format(float( sum( abs(ytrain-estimatedyincv)) / len(ytrain) )) )
# yyplot
plt.figure(figsize=figure.figaspect(1))
plt.scatter( ytrain, estimatedyincv)
YMax = np.max( np.array([np.array(ytrain), estimatedyincv]))
YMin = np.min( np.array([np.array(ytrain), estimatedyincv]))
plt.plot([YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin)], [YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin)], 'k-')
plt.ylim(YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin))
plt.xlim(YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin))
plt.xlabel("Actual Y")
plt.ylabel("Estimated Y in CV")
plt.show()

# Estimate y of test dataset
predictedytest = np.ndarray.flatten( regressionmodel.predict(std_Xtest) )
predictedytest = predictedytest*ytrain.std(ddof=1) + ytrain.mean()
# r2p, RMSEp, MAEp
print( "r2p: {0}".format(float( 1 - sum( (ytest-predictedytest )**2 ) / sum((ytest-ytest.mean())**2) )) )
print( "RMSEp: {0}".format(float( ( sum( (ytest-predictedytest)**2 )/ len(ytest))**0.5 )) )
print( "MAEp: {0}".format(float( sum( abs(ytest-predictedytest)) / len(ytest) )) )
# yyplot
plt.figure(figsize=figure.figaspect(1))
plt.scatter( ytest, predictedytest)
YMax = np.max( np.array([np.array(ytest), predictedytest]))
YMin = np.min( np.array([np.array(ytest), predictedytest]))
plt.plot([YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin)], [YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin)], 'k-')
plt.ylim(YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin))
plt.xlim(YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin))
plt.xlabel("Actual Y")
plt.ylabel("Estimated Y")
plt.show()

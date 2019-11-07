import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.datasets import load_boston

boston = load_boston()
df = DataFrame(boston.data,columns=boston.feature_names)
df['MEDV']=np.array(boston.target)

X = df.iloc[:, :-1].values
y = df.loc[:, 'MEDV'].values

from sklearn.cross_validation import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size=0.3,random_state=666)

from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor()
forest.fit(X_train,y_train)

y_train_pred=forest.predict(X_train)
y_test_pred=forest.predict(X_test)

from sklearn.metrics import mean_squared_error
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
from sklearn.metrics import r2_score
print('MSE train : %.3f, test : %.3f' % (r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))

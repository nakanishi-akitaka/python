# -*- coding: utf-8 -*-
"""
Scikit-learnによる多層パーセプトロン
https://data-science.gr.jp/implementation/iml_sklearn_mlp.html

Created on Thu Jan 10 12:33:38 2019

@author: Akitaka
"""

import sklearn
from sklearn.neural_network import MLPClassifier
import numpy as np
np.random.seed(0)
 
def main():
    # 1. reading data
    xtrain,ttrain=[],[]
    fin=open("classification_01_learning.txt","r")
    for i,line in enumerate(fin):
        line=line.rstrip()
        if line:
            tmp=line.split("\t")
            tmpx=tmp[0].split(",")
            tmpx=[float(j) for j in tmpx]
            tmpt=int(tmp[1])
            xtrain.append(tmpx)
            ttrain.append(tmpt)
    fin.close()
    xtrain=np.asarray(xtrain,dtype=np.float32)
    ttrain=np.asarray(ttrain,dtype=np.int32)
     
    # 2. learning, cross-validation
    diparameter={"hidden_layer_sizes":[(100,),(200,),(300,)],"max_iter":[1000],"batch_size":[20,50,100,200],"early_stopping":[True],"random_state":[123],}
    licv=sklearn.model_selection.GridSearchCV(MLPClassifier(),param_grid=diparameter,scoring="accuracy",cv=5)
    licv.fit(xtrain,ttrain)
    predictor=licv.best_estimator_
    sklearn.externals.joblib.dump(predictor,"predictor_mlp.pkl",compress=True)
     
    # 3. evaluating the performance of the predictor
    liprediction=predictor.predict(xtrain)
    table=sklearn.metrics.confusion_matrix(ttrain,liprediction)
    tn,fp,fn,tp=table[0][0],table[0][1],table[1][0],table[1][1]
    print("TPR\t{0:.3f}".format(tp/(tp+fn)))
    print("SPC\t{0:.3f}".format(tn/(tn+fp)))
    print("PPV\t{0:.3f}".format(tp/(tp+fp)))
    print("ACC\t{0:.3f}".format((tp+tn)/(tp+fp+fn+tn)))
    print("MCC\t{0:.3f}".format((tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(1/2)))
    print("F1\t{0:.3f}".format((2*tp)/(2*tp+fp+fn)))
     
    # 4. printing parameters of the predictor
    print(sorted(predictor.get_params(True).items()))
     
    # 5. printing importance of each attribute (connection weight)
    print(np.dot(predictor.coefs_[0],predictor.coefs_[1]).T)
     
if __name__ == '__main__':
    main()
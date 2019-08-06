# -*- coding: utf-8 -*-
"""
Scikit-learnによる多層パーセプトロン
https://data-science.gr.jp/implementation/iml_sklearn_mlp.html

Created on Thu Jan 10 12:34:48 2019

@author: Akitaka
"""

import sklearn
import numpy as np
np.random.seed(0)
 
def main():
    # 1. reading data
    xtest,ttest=[],[]
    fin=open("classification_01_test.txt","r")
    for i,line in enumerate(fin):
        line=line.rstrip()
        if line:
            tmp=line.split("\t")
            tmpx=tmp[0].split(",")
            tmpx=[float(j) for j in tmpx]
            tmpt=int(tmp[1])
            xtest.append(tmpx)
            ttest.append(tmpt)
    fin.close()
    xtest=np.asarray(xtest,dtype=np.float32)
    ttest=np.asarray(ttest,dtype=np.int32)
     
    # 2. reading predictor
    predictor=sklearn.externals.joblib.load("predictor_mlp.pkl")
     
    # 3. evaluating the performance of the predictor on the test dataset
    liprediction=predictor.predict(xtest)
    table=sklearn.metrics.confusion_matrix(ttest,liprediction)
    tn,fp,fn,tp=table[0][0],table[0][1],table[1][0],table[1][1]
    print("TPR\t{0:.3f}".format(tp/(tp+fn)))
    print("SPC\t{0:.3f}".format(tn/(tn+fp)))
    print("PPV\t{0:.3f}".format(tp/(tp+fp)))
    print("ACC\t{0:.3f}".format((tp+tn)/(tp+fp+fn+tn)))
    print("MCC\t{0:.3f}".format((tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(1/2)))
    print("F1\t{0:.3f}".format((2*tp)/(2*tp+fp+fn)))
 
if __name__ == '__main__':
    main()
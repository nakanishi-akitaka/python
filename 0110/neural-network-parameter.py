# -*- coding: utf-8 -*-
"""
ニューラルネットワークのパラメータ設定方法(scikit-learnのMLPClassifier)
https://spjai.com/neural-network-parameter/

Created on Thu Jan 10 14:49:21 2019

@author: Akitaka
"""


from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def main():
    digits = datasets.load_digits()
    data = digits.data
    target = digits.target
    #cross_validation
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=0)
    # NN learning
    clf = MLPClassifier()
    # lerning
    clf.fit(data_train, target_train)
    # preedict test data
    predict = clf.predict(data_test)
    # checking answer
    print(classification_report(target_test, predict))

if __name__ == '__main__':
    main()

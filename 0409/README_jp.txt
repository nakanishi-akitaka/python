# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 23:54:24 2019

@author: Akitaka
"""

[1a2]
[デモのプログラムあり] 勾配ブースティングGradient Boosting、
特に Gradient Boosting Decision Tree (GBDT), XGBoost, LightGBM 
https://datachemeng.com/gradient_boosting/
勾配ブースティング (Gradient Boosting) とは？
    アンサンブル学習の一つ
    ブースティングの一つ
    クラス分類でも回帰でも可能
    クラス分類手法・回帰分析手法は何でもよいが、基本的に決定木を用いる
        Gradient Boosting Decision Tree (GBDT)
    クラス分類モデルで誤って分類されてしまったサンプルや、
        回帰モデルで誤差の大きかったサンプルを、改善するように (損失関数の値が小さくなるように) 
        新たなモデルが構築される
"""
つまりは、決定木+ブースティング? or 「勾配」ブースティングという特殊なもの？
=> 
勾配ブースティング = 勾配降下法を使ったブースティング
特に、決定木を使ったものがよくつ使われている
それは本来は、Gradient Boosting Decision Tree (GBDT)と呼ぶべきもの
ref:
    https://analytics-and-intelligence.net/archives/678
    勾配ブースティング
    勾配降下法を使ったブースティングのこと。正直なところアダブーストとの違いがいまいちよくわからない。
    GBDT(Gradient Boosting Decision Tree)
    弱学習機に決定木を使った勾配ブースティングのこと。かっこよく聞こえる。    
    
    http://rautaku.hatenablog.com/entry/2018/01/13/190818
    まず勾配ブースティングとは複数の弱学習器を組み合わせたアンサンブル学習の一種で、
    その中でも1つずつ順番に弱学習器を構築していく手法です。 
    一つずつ順番作っていくので、新しい弱学習器を作るときには、バギングと呼ばれるすべての弱学習器を
    独立に学習する方法と比べて並列計算ができず時間がかかります。時間はかかるのですが、
    過学習（高バリアンス）と学習不足（高バイアス）のバランスをうまく取ることができます。
    イメージとしては、ブースティングは最適化、バギングは平均化。　　

    https://www.st-hakky-blog.com/entry/2017/08/08/092031
    一般的に誤差関数というものに対して、それを最小にするようなパラメーターを持つ学習器を
    構築することは困難です。そこで、観測データに従って以下のような勾配をまず計算します。
    この勾配に対して $t$ ステップ目の弱学習器がフィットするように学習を行います。
    つまり、やっていることとしては以下の式のように、勾配と弱学習機の誤差が最小になるように、
    弱学習器を学習することをしています。
    # このあたりが、勾配降下法？

scikit-learnにプログラムあり
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

XGBoost (eXtreme Gradient Boosting)
https://xgboost.readthedocs.io/en/latest/python/python_intro.html
LightGBM (Light Gradient Boosting Model)
https://lightgbm.readthedocs.io/en/latest/
この2つは、オーバーフィッティングを防いだり、計算速度を上げるための工夫あり

XGBoostの工夫
決定木を構築するときの評価関数Eに以下の式を加える
    γ * T + λ/2 ||w||**2
    T:最終ノードの数
    w:すべての葉ノードの値が格納されたベクトル
    γ、λ:ハイパーパラメータ
    これを小さくする => ノード数・モデルの更新を小さくする => オーバーフィッティングの防止

LightGBMの主な特徴
    m回目の学習の時、勾配の絶対値の大きなサンプルa% + それ以外のサンプルb%を、
    勾配を(1-a/b)倍して用いる

    0でない値を持つサンプルの被りが少ない変数は、一緒にしてバンドルとして扱う
    変数からではなく、バンドルから決定木を作成
    ヒストグラムで値をざっくり分けてから決定木の分岐を探索
    => オーバーフィッティングの防止 & 計算速度向上

サンプルプログラム
https://github.com/hkaneko1985/gradient_boosting
また回帰分析やクラス分類において、scikit-learn の GBDT (回帰分析・クラス分類)、XGBoost、LightGBM 
のデモンストレーションのプログラムも準備しました。
さらに、optuna を利用して、クロスバリデーションの結果をベイズ最適化することで、
各モデルのハイパーパラメータを最適化するデモンストレーションのプログラムもあります。
ただ、デフォルトのパラメータでも良好なモデルを構築できたり、
ベイズ最適化するよりテストデータの推定結果がよくなったりすることもあります。
デフォルトのパラメータとベイズ最適化したパラメータとで両方比較するとよいです。


optunaについて
https://optuna.org/
$ pip install optuna


参考文献
Natekin, A. Knoll, 
"Gradient boosting machines", 
a tutorial, Front. Neurobot., 7, 1-21, 2013
https://doi.org/10.3389/fnbot.2013.00021

Hastie, R. Tibshirani, J. Friedman, 
"The Elements of Statistical Learning: Data mining, Inference, and Prediction", 
Second Edition, Springer, 2009
https://web.stanford.edu/~hastie/ElemStatLearn/

Chen, C. Guestrin, 
"XGBoost: A Scalable Tree Boosting System", 
arXiv:1603.02754, 2016
https://arxiv.org/abs/1603.02754

Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T. Y. Liu, 
"LightGBM: A Highly Efficient Gradient Boosting Decision Tree", 
NIPS Proceedings, 2017
https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
"""

この記事について
https://twitter.com/hirokaneko226/status/1115372759305965568
[デモのプログラムあり] 勾配ブースティングGradient Boosting、
特に Gradient Boosting Decision Tree (GBDT), XGBoost, LightGBM

https://twitter.com/steroidinlondon/status/1115398902482493440
サンプルコードを拝見しましたが、n_estimators も調節すると良いかと思います。
ランダムフォレストと異なり、木の本数が多いとオーバーフィッティングする可能性が出てきますので。




demo_of_gradient_boosting_classification_default_hyperparam.py
※デフォルトのハイパーパラメータ
training samples
[0, 1, 2]
[[34  0  0]
 [ 0 32  0]
 [ 0  0 39]]
training samples in CV
[0, 1, 2]
[[34  0  0]
 [ 0 30  2]
 [ 0  4 35]]

test samples
[0, 1, 2]
[[16  0  0]
 [ 0 17  1]
 [ 0  0 11]]


demo_of_gradient_boosting_classification_with_optuna.py
※ optuna でハイパーパラメータ最適化 
training samples
[0, 1, 2]
[[34  0  0]
 [ 0 32  0]
 [ 0  0 39]]
training samples in CV
[0, 1, 2]
[[34  0  0]
 [ 0 30  2]
 [ 0  3 36]]

test samples
[0, 1, 2]
[[16  0  0]
 [ 0 17  1]
 [ 0  0 11]]
 

demo_of_gradient_boosting_regression_default_hyperparam.py
※デフォルトのハイパーパラメータ
￼
r2: 0.9863077061180023
RMSE: 1.0894128454370786
MAE: 0.85499103537567

￼
r2cv: 0.8452083493148206
RMSEcv: 3.6629260560216315
MAEcv: 2.3338142217046

￼
r2p: 0.8541466528646792
RMSEp: 3.4393274086141568
MAEp: 2.2786494962498396


demo_of_gradient_boosting_regression_with_optuna.py
※ optuna でハイパーパラメータ最適化 
r2: 0.9904202433995527
RMSE: 0.9112375825271617
MAE: 0.6197253517749376

￼
r2cv: 0.8322655345168288
RMSEcv: 3.8129888943103953
MAEcv: 2.5077317326283817

￼
r2p: 0.8187721486237058
RMSEp: 3.833785331798006
MAEp: 2.3696366770084145

デフォルトでも最適化でも、大差ない



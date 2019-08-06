# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 17:17:59 2019

@author: Akitaka
"""
[1] 機械学習
[1a] サイトで勉強　金子研究室
データ解析に関するいろいろな手法・考え方のまとめ
https://datachemeng.com/summarydataanalysis/

[1b] + [2a] scikit learnでニューラルネットワーク
[todo]->[doing] とりあえず、NNの計算をテストしてみる

[1b1] 以前にもやった。復習。
以下、使用プログラム一覧

20180619 all_regression_models.py test1_16rgr.py
    クラス分類手法を総当たりで試すときに、一応使った程度

20180620 test1_17rgr.py, test2_17rgr_TMXO2_gap.py, test3_17rgr_AdaBoost.py
    クラス分類手法を総当たりで試すときに、一応使った程度

20180710
    test3_DL.py
        scikit-learnのディープラーニング実装簡単すぎワロタ
        http://aiweeklynews.com/archives/50172518.html
        サンプルにかいてある、csvファイルが用意されてないので、書き換え
        しかし、学習がうまく収束しない！原因は不明
    -> 今回しても、同様にエラー

    test5_DL.py
    http://scikit-learn.org/stable/auto_examples/neural_networks/
        plot_mlp_training_curves.html
    -> 成功

    test6_DL.py
    http://scikit-learn.org/stable/auto_examples/neural_networks/
        plot_mlp_alpha.html
    -> 成功

    test7_DL.py
    http://scikit-learn.org/stable/auto_examples/classification/
        plot_classifier_comparison.html
    -> 成功

20180716　test3_clf.py
    クラス分類手法を総当たりで試すときに、一応使った程度

他、NNをLDAで代用(?)しているのはあった




[1b2] 新しく検索したサイトのサンプルなど

Scikit-learnによる多層パーセプトロン
https://data-science.gr.jp/implementation/iml_sklearn_mlp.html
    多層パーセプトロン (multilayer perceptron (MLP)) は
    数あるニューラルネットワークの学習法の中でも最も基本的な手法．
    Scikit-learn に実装されている MLP は
    Chainer や Theano 等で実現できるような高度なネットワークを構築することはできないが，
    専用のライブラリと比較して手軽に MLP を実装できるという利点がある．
    ただの MLP ではあるが，中間層の数，すわなち深さも自由に変更することができ，
    一応，ディープラーニングもできる．

    プログラム１：学習による予測器の生成
    1.データの読み込み．
    2.グリッドサーチおよびクロスバリデーションによる学習．
    3.構築した予測器のラーニングデータセットにおける性能の評価結果の出力．
    4.予測器のパラメーターの出力．
    5.入力ベクトルの各特徴量の重要度の出力．
    →成功
    iml_sklearn_mlp1.py
    学習用データ400個なので、Tcのと大差ない

    プログラム２：構築した予測器を用いたテスト
    iml_sklearn_mlp2.py
    →成功

    学習パラメーター一覧
        日本語で解説しているのでわかり易い
        (1層目のノード数, 2層目のノード数,...)のデフォルト = (100,) = 100ノード1層のみ

[?] ニューラルネットワーク, 多層パーセプトロン, ディープラーニングとはどういう違い？包含関係？
    現状の理解
    MLPもDLも、NNの一種。MLPはDLの一種。
    ニューラルネットワーク
        順伝播型ニューラルネットワーク
            単純パーセプトロン
            多層パーセプトロン
                入力層と隠れ層と出力層が全結合である、もっとも単純なディープラーニング
                狭義には4層以上
        畳み込みニューラルネットワーク
                畳み込み層とプーリング層をもつディープラーニング
        再帰型ニューラルネットワーク
                時系列データ、前後のデータに意味のあるデータに対して、
                その特徴を学習できるディープラーニング
        etc...

    ref:
    https://qiita.com/ishizakiiii/items/4ea799a47f70a45b687c
    https://qiita.com/begaborn/items/3fb064215d7f6e6cce96
    http://hokuts.com/2016/12/13/cnn1/


Python(scikit-learn)でニューラルネットワーク
https://qiita.com/tsal3290s/items/3c0b8713a26ee10b689e
    超シンプルなAPIでニューラルネットワークが利用できるようになった。
    サンプルコードではアヤメ分類を行っている。
    sklearn_nn.py
    -> 成功
        いつも通りの手順(train_test_split, fitなど。)
        パラメータ調整すらない。最低限必要なことのみの計算。

1.17。 ニューラルネットワークモデル（監視対象）
https://code.i-harness.com/ja/docs/scikit_learn/
    modules/neural_networks_supervised
    """
    恐らくは、公式ユーザーガイドの機械翻訳
    https://scikit-learn.org/0.18/modules/neural_networks_supervised.html
    以前に見た、下のよりも広い範囲が日本語化されてる
        【翻訳】scikit-learn 0.18 User Guide 目次
        https://qiita.com/nazoking@github/items/267f2371757516f8c168
    サンプルコードは短い。上のアヤメ分類＋αぐらい。
    """

3.3 ニューラルネットワーク(ディープラーニング)
https://nozma.github.io/ml_with_python_note/
    3-3-ニューラルネットワークディープラーニング.html
    """
    NNの基礎から詳しくわかり易い
    特に、パラメータ変更時の結果が可視化されているので、パラメータの意味が分かりやすい
    他、特徴量ごとの重要性を可視化

    NNの特徴
    大量のデータを使って非常に複雑なモデルを構築できること。
    時間とデータを費やし十分にパラメータ調整を行えば回帰でも分類でも
    他のアルゴリズムに勝てる可能性がある。
    訓練には時間がかかる。
    それぞれの特徴量のスケールが近くないと上手く動かない。
    パラメータのチューニングはそれ自体が技術となる程度に複雑で奥が深い。

    33nndl.py
    -> mglearn をインストール
        ※ conda installは不可能
        https://venuschjp.blogspot.com/2018/01/python-1.html
    -> 成功
    """

ニューラルネットワークのパラメータ設定方法(scikit-learnのMLPClassifier)
https://spjai.com/neural-network-parameter/
    """
    合計21個のパラメータについて詳しく解説
    solverが～の時に必要、といった、パラメータもあるので、21個を一度に全部調整することはない
        activation='relu', solver='adam' (どちらもdefault) がオススメらしい
    サンプルプログラムは、パラメータ固定。手動で書き換えて調整しよう、とのこと。
    neural-network-parameter.py
    """

以降のサイトは、解説そこそこ詳しい。サンプルプログラムがシンプル(パラメータ調整なし)
<div>
【ニューラルネットワーク】基本原理と単純パーセプトロンの学習計算
https://algorithm.joho.info/machine-learning/neural-network/
    単純パーセプトロンでは、教師データを与えて勾配降下法（最急降下法）により、重みを決定します。

【ニューラルネット】多層パーセプトロン（MLP）の原理・計算式
https://algorithm.joho.info/machine-learning/neural-network-mlp/
    多層パーセプトロンでは、「確率的勾配降下法」で重みを更新します。
    誤差逆伝播により、重みの更新に必要な量を計算します。

【CNN】畳み込みニューラルネットワークの原理・仕組み
https://algorithm.joho.info/machine-learning/convolutional-neural-network/
    ディープラーニングの手法の１つで、主に画像認識に利用されています。
    ニューラルネットワークの重み計算に「畳み込み演算（Convolution）」が用いられている

【Python】単純パーセプトロンの重み計算（勾配降下法・最急降下法）
https://algorithm.joho.info/programming/
    python/simple-perceptron-gradient-descent-py/
    機械学習用のライブラリーはなしで、一から作ってる


【Scikit-learn】ニューラルネットワーク（多層パーセプトロン・MLP）
https://algorithm.joho.info/machine-learning/
    python-scikit-learn-neural-network-predict/
    """
    python-scikit-learn-neural-network-predict.py
    -> .as_matrix() -> .valuesに変更　※使われなくなるため
    -> 成功
    """

【Scikit-learn】ニューラルネットワークの識別率を計算
https://algorithm.joho.info/machine-learning/
    python-scikit-learn-neural-network-rate/
    """
    python-scikit-learn-neural-network-rate.py
    -> 成功
    NNに限ったことでもないので、やらなくてもよかった
    """

【Scikit-learn】ニューラルネットワーク学習モデルのファイル出力・保存
https://algorithm.joho.info/machine-learning/
    python-scikit-learn-neural-network-dump/
    """
    python-scikit-learn-neural-network-dump.py
    ->成功
    NNに限ったことでもないので、やらなくてもよかった
    """

【Scikit-learn】ニューラルネットワーク学習モデルを読み込む(インポート)
https://algorithm.joho.info/machine-learning/
    python-scikit-learn-neural-network-import/
    """
    python-scikit-learn-neural-network-import.py
    ->成功
    NNに限ったことでもないので、やらなくてもよかった
    """

【Scikit-learn】ニューラルネットワークでアヤメ分類（MLP）
https://algorithm.joho.info/machine-learning/
    python-scikit-learn-neural-network-iris/
    """
    python-scikit-learn-neural-network-iris.py
    ->成功
    """
</div>

pythonでニューラルネットワーク実装
https://qiita.com/ta-ka/items/bcdfd2d9903146c51dcb
    機械学習用のライブラリーはなしで、一から作ってる！

いろいろやったが、一番参考になりそうなのは下か。
というより、唯一ハイパーパラメータをまともに(グリッドサーチで)調整している。
https://data-science.gr.jp/implementation/iml_sklearn_mlp.html
他、ハイパーパラメータの解説
https://spjai.com/neural-network-parameter/
https://nozma.github.io/ml_with_python_note/
    3-3-ニューラルネットワークディープラーニング.html



[1b3] 多層パーセプトロンで水素化物のTc予測プログラム
    tc_mlp.py
    パラメータの探索範囲を、上の一番参考になりそうなサイトの値を流用して、走らせた結果
    ※max_iterは必要に応じて増やした(以降も同様)
    Best parameters set found on development set:
    {'batch_size': 200, 'early_stopping': True, 'hidden_layer_sizes': (300,),
    'max_iter': 2000, 'random_state': 123}
    C:  RMSE, MAE, R^2 = 45.441, 32.389,  0.383
    CV: RMSE, MAE, R^2 = 46.589, 32.964,  0.351
    TST:RMSE, MAE, R^2 = 42.864, 31.329,  0.424
    177.98 seconds

    探索範囲を狭く "batch_size":[20,50,100,200] -> ['auto'],
    Best parameters set found on development set:
    {'batch_size': 'auto', 'early_stopping': True,
    'hidden_layer_sizes': (200,), 'max_iter': 3000, 'random_state': 123}
    C:  RMSE, MAE, R^2 = 43.498, 29.932,  0.408
    CV: RMSE, MAE, R^2 = 49.140, 34.685,  0.244
    TST:RMSE, MAE, R^2 = 46.193, 34.488,  0.436
    Predicted Tc is written in file tc_mlp.csv
    77.48 seconds

    "early_stopping":[True] -> del
    max_iterが足りない警告が頻発したので停止

    "random_state":[123] -> [42]
    Best parameters set found on development set:
    {'batch_size': 'auto', 'early_stopping': True,
    'hidden_layer_sizes': (300,), 'max_iter': 3000, 'random_state': 42}
    C:  RMSE, MAE, R^2 = 37.552, 26.359,  0.557
    CV: RMSE, MAE, R^2 = 43.372, 29.734,  0.410
    TST:RMSE, MAE, R^2 = 44.776, 29.603,  0.476
    102.96 seconds

    "hidden_layer_sizes":[(100,),(200,),(300,)] -> [(10,),(20,),(30,)]
    Best parameters set found on development set:
    {'batch_size': 'auto', 'early_stopping': True,
    'hidden_layer_sizes': (30,), 'max_iter': 3000, 'random_state': 42}
    C:  RMSE, MAE, R^2 = 45.507, 32.263,  0.407
    CV: RMSE, MAE, R^2 = 46.684, 33.162,  0.376
    TST:RMSE, MAE, R^2 = 38.451, 28.385,  0.429
    91.55 seconds

    ... -> :[(30,),(30,30,),(30,30,30,),(30,30,30,30,)]
    Best parameters set found on development set:
    {'batch_size': 'auto', 'early_stopping': True,
    'hidden_layer_sizes': (30, 30, 30), 'max_iter': 3000, 'random_state': 42}
    C:  RMSE, MAE, R^2 = 41.949, 29.814,  0.478
    CV: RMSE, MAE, R^2 = 42.967, 31.081,  0.452
    TST:RMSE, MAE, R^2 = 51.353, 35.856,  0.155
    60.44 seconds
    精度がいい訳でもないのに、過学習？


なぜうまくいかない？
変数の数が少ないせいか？
しかし、アヤメは変数の数少ないが上手くいっている
クラス分類とは話が違うかもしれないが、参考にできないか？
    https://qiita.com/tsal3290s/items/3c0b8713a26ee10b689e
        アヤメ
        clf = MLPClassifier(solver="sgd",random_state=0,max_iter=10000)
    https://algorithm.joho.info/machine-learning/
        python-scikit-learn-neural-network-iris/
        アヤメ
        clf = MLPClassifier(solver="sgd",random_state=0,max_iter=10000)
    https://spjai.com/neural-network-parameter/
        デジタル画像
        clf = MLPClassifier()

    https://nozma.github.io/ml_with_python_note/
        3-3-ニューラルネットワークディープラーニング.html
        moon
        mlp = MLPClassifier(solver='lbfgs', random_state=0)
        mlp = MLPClassifier(solver='lbfgs', random_state=0,
            hidden_layer_sizes=[10])
        mlp = MLPClassifier(solver='lbfgs', random_state=0,
            hidden_layer_sizes=[10, 10])
        mlp = MLPClassifier(solver='lbfgs', activation='tanh',
            random_state=0, hidden_layer_sizes=[10, 10])
        mlp = MLPClassifier(solver='lbfgs', random_state=0,
            hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],　alpha=alpha)
        mlp = MLPClassifier(solver='lbfgs', random_state=i,
            hidden_layer_sizes=[100, 100])

意外とsgdを使っている。adamで大体OKであると、書いてあったのに。
また、データサイズが小さいとノードの数を減らすらしい
->
    "hidden_layer_sizes":[(100,),(10,),(10,10,),(10,10,10,)],
    "solver":['lbfgs', 'sgd', 'adam'],

    Best parameters set found on development set:
    {'batch_size': 'auto', 'early_stopping': True,
    'hidden_layer_sizes': (10, 10, 10), 'max_iter': 10000,
    'random_state': 42, 'solver': 'lbfgs'}
    C:  RMSE, MAE, R^2 = 18.880, 13.057,  0.874
    CV: RMSE, MAE, R^2 = 32.554, 21.974,  0.626
    TST:RMSE, MAE, R^2 = 47.108, 26.872,  0.567
    209.40 seconds
明らかに過学習だが、だいぶマシにはなった。
adam以外も良い事があるらしい
また、ノード数は大きいほどいい訳でもないと分かる


[todo] アルファインパクトでのニューラルネットワークを読む
https://alphaimpact.jp/2017/05/04/perceptron/
https://alphaimpact.jp/2017/05/11/multilayer-perceptron/
https://alphaimpact.jp/2017/05/18/neuralnetwork-feature-extraction/    
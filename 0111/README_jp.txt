# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 17:17:59 2019

@author: Akitaka
"""

[1] 機械学習
[1a] サイトで勉強　金子研究室
データ解析に関するいろいろな手法・考え方のまとめ
https://datachemeng.com/summarydataanalysis/

[1b1]+ [2a] 多層パーセプトロンで水素化物のTc予測プログラム
    tc_mlp.py

以下の２つで、ハイパーパラメータの調整すべきポイントが少し変わっている！
改めて読み直し
https://data-science.gr.jp/implementation/iml_sklearn_mlp.html
https://spjai.com/neural-network-parameter/
＋公式ドキュメントも見直す
https://scikit-learn.org/stable/modules/generated/
    sklearn.neural_network.MLPRegressor.html

値はデフォルト
"hidden_layer_sizes":(100,)   共通して要調整
"activation":'relu'           共通して'relu'でOK
"solver":'adam'
    共通して'adam'でOK。サンプルではsgdもあり
    後者
    'blfgs'は、1000以下の小さいデータセットの場合に高パフォーマンス
"alpha":0.0001
    後者より
    一般的に0.01 ~ 0.00001 の間の値
    過学習が考えられるような出力結果が得られた場合や、
    ネットワークの自由度が高い場合等はこのパラメータを上げてみると良い
"batch_size":min(200, n_samples)
    solver='sgd','adam'のみ有効
    前者：グリッドサーチした方が良い
    後者：基本はデフォルトの値で問題無い
"learning_rate_init":0.001
    solver='sgd','adam'のみ有効
    共通して要調整
"learning_rate":'constant'
    solver='sgd'のみ有効
    共通して要調整
"power_t":0.5                 共通して不明
    solver='sgd'のみ有効
"max_iter":200
    後者
    少なすぎるとうまく学習できない、多すぎると過学習
"shuffle":True
    solver='sgd','adam'のみ有効
    共通してTrue
"random_state":np.ramdom,
    前者：整数で固定するべき
"tol":1e-4
    learning_rate='adaptive'以外で有効
    後者：大きくしすぎると学習途中で終了してしまいますが、小さくしすぎると過学習が起こりやすくなります。
"verbose":False               特になし(画面に進捗状況を表示するかどうかだけの問題)
"warm_start":False,           特になし
"momentum":0.9,
    solver='sgd'のみ有効
    特になし
"nesterovs_momentum":True,
    solver='sgd'かつ momentum>0 のみ有効
    後者「基本はTrue」
"early_stopping":False,
    solver='sgd','adam'のみ有効
    後者「epoch数を多く設定している場合や、トレーニングデータが十分にある場合などは使うとよい」
"validation_fraction":0.1,
    early_stopping=True のみ有効
    特になし
"beta_1":0.9
    solver='adam'のみ有効
    共通してデフォルト推奨
"beta_2":0.999
    solver='adam'のみ有効
    共通してデフォルト推奨
"epsilon":1e-9
    solver='adam'のみ有効
    後者はデフォルト推奨

よって、データ数 < 1000であることから、
"solver"='lbfgs'としてしまえば、調整するべきパラメータは少ない
"hidden_layer_sizes"
"alpha"
"max_iter"
"tol"

意外と、max_iterでも性能がかなり変わる & 時間がかからない
説明通り、多すぎると過学習が起きる
max_iter = 1000 -> 500 -> 300 -> 200　-> 100
C:  RMSE, MAE, R^2 = 15.271,  9.603,  0.929
CV: RMSE, MAE, R^2 = 34.845, 22.486,  0.629
TST:RMSE, MAE, R^2 = 82.575, 39.165, -0.965
389.90 seconds
->
C:  RMSE, MAE, R^2 = 20.936, 13.294,  0.872
CV: RMSE, MAE, R^2 = 41.751, 24.830,  0.492
TST:RMSE, MAE, R^2 = 31.617, 21.560,  0.650
165.60 seconds
->
C:  RMSE, MAE, R^2 = 22.985, 15.113,  0.856
CV: RMSE, MAE, R^2 = 46.325, 27.692,  0.415
TST:RMSE, MAE, R^2 = 24.424, 17.732,  0.689
95.74 seconds
->
C:  RMSE, MAE, R^2 = 23.495, 16.133,  0.821
CV: RMSE, MAE, R^2 = 43.888, 26.133,  0.375
TST:RMSE, MAE, R^2 = 35.272, 23.093,  0.702
85.06 seconds
->
C:  RMSE, MAE, R^2 = 28.603, 19.735,  0.768
CV: RMSE, MAE, R^2 = 39.172, 27.063,  0.564
TST:RMSE, MAE, R^2 = 32.821, 23.484,  0.557
34.34 seconds

また、クロスバリデーションの数を3->5と増やすと、C,CV,TSTの差が小さくなる & 時間はかかる
C:  RMSE, MAE, R^2 = 24.489, 16.210,  0.819
CV: RMSE, MAE, R^2 = 32.417, 22.589,  0.683
TST:RMSE, MAE, R^2 = 30.951, 19.473,  0.712
181.56 seconds

隠れ層について、[(10,),(10,10,),(10,10,10,),(10,10,10,10,)]のように
ノードは固定して、何層かを最適化させる　※max 4層

(5, 5,)
C:  RMSE, MAE, R^2 = 36.575, 26.800,  0.618
CV: RMSE, MAE, R^2 = 41.045, 28.951,  0.518
TST:RMSE, MAE, R^2 = 32.439, 24.689,  0.593
304.89 seconds

(10, 10,)
C:  RMSE, MAE, R^2 = 31.796, 22.017,  0.701
CV: RMSE, MAE, R^2 = 39.296, 25.999,  0.543
TST:RMSE, MAE, R^2 = 28.159, 20.249,  0.738
188.20 seconds

(20, 20, 20, 20,)
C:  RMSE, MAE, R^2 = 25.413, 19.487,  0.790
CV: RMSE, MAE, R^2 = 37.207, 26.447,  0.550
TST:RMSE, MAE, R^2 = 38.141, 22.233,  0.659
260.96 seconds

(40, 40, )
C:  RMSE, MAE, R^2 = 21.096, 14.509,  0.858
CV: RMSE, MAE, R^2 = 35.357, 23.311,  0.601
TST:RMSE, MAE, R^2 = 29.436, 20.301,  0.784
421.33 seconds

(50, 50, )
C:  RMSE, MAE, R^2 = 17.847, 12.788,  0.898
CV: RMSE, MAE, R^2 = 35.270, 22.792,  0.601
TST:RMSE, MAE, R^2 = 32.797, 22.198,  0.738
918.87 seconds




[1b2] アルファインパクトでのニューラルネットワークを読む
第11回 競馬で学ぶニューラルネットワーク 〜パーセプトロン編〜
https://alphaimpact.jp/2017/05/04/perceptron/

第12回 競馬で学ぶニューラルネットワーク 〜多層パーセプトロン編〜
https://alphaimpact.jp/2017/05/11/multilayer-perceptron/
    隠れ層の数 = 1, ノードの数 = 1-14 ※ 特徴量の数 = 14
    と少ない割には、うまくいっている

第13回 競馬で学ぶニューラルネットワーク 〜特徴抽出編〜
https://alphaimpact.jp/2017/05/18/neuralnetwork-feature-extraction/




[2] 学生
[2a] 古川君
2019年の方針 ref:20181228
    1.卒論は回帰分析手法の方向
    2.来週は次週、再来週以降は直接会って中西が解説
01/11 14:50-16:15
15:10 - 16:10
サポートベクター回帰の説明
[todo]->[done] 回帰分析手法を決める ref:20190109
    1.kNN + アンサンブル
    2.ニューラルネットワーク、多層パーセプトロン、DL
    3.ガウス過程回帰、ベイズ最適化
一通り説明した後、古川君は2.を選ぶ
その後、忙しさなどを聞いて、「DLやるなら、かなり頑張る必要あり」という話をしていたら、3.に変更した
    仕事は最大3時間ぐらい
    DLは余裕ができたらやることに
    誘導したみたいになってしまった？反省
        古川君は「必要な情報を教えてもらった」と言ってくれたが
    ただ、1日30分話して終わり、というようなペースでは厳しいことも確かなのだが
    土日月の間に、進められるところは、進める
        スライド、論文、プログラムの練習
        このように、一人でも進められる課題をきちんと与えておかなかったことも反省材料

[todo] ガウス過程回帰の説明、計算について過去ノート見直し
20181107 ガウス過程回帰による水素化物Tc予測

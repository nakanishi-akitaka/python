# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 17:17:59 2019

@author: Akitaka
"""

[1] 機械学習
[1a] サイトで勉強　金子研究室
データ解析に関するいろいろな手法・考え方のまとめ
https://datachemeng.com/summarydataanalysis/

[1a2]
【機械学習】OOB (Out-Of-Bag) とその比率
https://qiita.com/kenmatsu4/items/1152d6e5634921d9246e
    ランダムフォレストで使う、ブートストラップサンプリングのOOB(Out-Of-Bag)
    = N個から重複ありでN個選ぶときに選ばれなかったもの
    よって、あるサンプルが選ばれない確率は[(N-1)/N]**N → 1/e (N → ∞) ~ 36%

[1a3]
超伝導転移温度の推移 - 橘高 俊一郎 ＠東京大学物性研究所　榊原研究室
http://sakaki.issp.u-tokyo.ac.jp/user/kittaka/contents/others/tc-history.html
超伝導転移温度の記録の変化をグラフにしました。(Made by S. Kitagawa and S. Kittaka)

ref
昨日(20190116)の清水研究室用の画像

[1a4] ニューラルネットワーク(多層パーセプトロン)で水素化物のTc予測プログラム
 + パラメータの調整 + 特徴量の変更
    tc_mlp.py

ref:
20190110ノート
20190111ノート
https://data-science.gr.jp/implementation/iml_sklearn_mlp.html
https://spjai.com/neural-network-parameter/
https://scikit-learn.org/stable/modules/generated/
    sklearn.neural_network.MLPRegressor.html

"hidden_layer_sizes"のみを調整。"solver"='lbfgs'、他はデフォルト

以下、探索範囲＋モデルの制度評価
hidden_layer_sizes only optimization
"hidden_layer_sizes":[(1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,)],
    'hidden_layer_sizes': (9,)
    C:  RMSE, MAE, R^2 = 32.106, 23.635,  0.693
    CV: RMSE, MAE, R^2 = 41.901, 29.646,  0.478
    TST:RMSE, MAE, R^2 = 35.957, 25.711,  0.589
    13.42 seconds

"hidden_layer_sizes":[(10,),(20,),(40,),(60,),(80,),(100,)],
    'hidden_layer_sizes': (40,)
    C:  RMSE, MAE, R^2 = 27.640, 19.136,  0.773
    CV: RMSE, MAE, R^2 = 37.626, 24.490,  0.579
    TST:RMSE, MAE, R^2 = 35.311, 22.737,  0.604
    15.30 seconds

"hidden_layer_sizes":[(100,),(200,),(400,),(600,),(800,),(1000,)],
    'hidden_layer_sizes': (400,)
    C:  RMSE, MAE, R^2 = 22.427, 14.901,  0.850
    CV: RMSE, MAE, R^2 = 49.679, 28.062,  0.266
    TST:RMSE, MAE, R^2 = 33.078, 21.065,  0.652
    137.54 seconds

"hidden_layer_sizes":[(1,1),(2,2),(4,4),(6,6),(8,8),(10,10)],
    'hidden_layer_sizes': (8, 8)
    C:  RMSE, MAE, R^2 = 32.330, 21.817,  0.689
    CV: RMSE, MAE, R^2 = 37.768, 25.701,  0.576
    TST:RMSE, MAE, R^2 = 35.927, 24.767,  0.590
    11.17 seconds

"hidden_layer_sizes":[(10,10),(20,20),(40,40),(60,60),(80,80),(100,100)],
    'hidden_layer_sizes': (100, 100)
    C:  RMSE, MAE, R^2 = 19.889, 13.086,  0.882
    CV: RMSE, MAE, R^2 = 40.848, 24.543,  0.503
    TST:RMSE, MAE, R^2 = 30.266, 19.086,  0.709
    41.21 seconds

"hidden_layer_sizes":[(100,100),(200,200),(400,400),
    (600,600),(800,800),(1000,1000)],
    'hidden_layer_sizes': (100, 100)
    C:  RMSE, MAE, R^2 = 19.889, 13.086,  0.882
    CV: RMSE, MAE, R^2 = 39.193, 22.143,  0.543
    TST:RMSE, MAE, R^2 = 30.266, 19.086,  0.709
    2091.48 seconds

"hidden_layer_sizes":[(1,1,1),(2,2,2),(4,4,4),(6,6,6),(8,8,8),(10,10,10)],
    'hidden_layer_sizes': (8, 8, 8)
    C:  RMSE, MAE, R^2 = 28.435, 20.281,  0.759
    CV: RMSE, MAE, R^2 = 37.523, 24.776,  0.581
    TST:RMSE, MAE, R^2 = 36.600, 23.993,  0.574
    12.83 seconds

"hidden_layer_sizes":[(10,10,10),(20,20,20),(40,40,40),
    (60,60,60),(80,80,80),(100,100,100)],
    'hidden_layer_sizes': (40, 40, 40)
    C:  RMSE, MAE, R^2 = 20.776, 13.770,  0.872
    CV: RMSE, MAE, R^2 = 37.154, 24.075,  0.589
    TST:RMSE, MAE, R^2 = 30.117, 19.244,  0.712
    60.57 seconds

"hidden_layer_sizes":[(1,1,1,1),(2,2,2,2),(4,4,4,4),
    (6,6,6,6),(8,8,8,8),(10,10,10,10)],
    'hidden_layer_sizes': (10, 10, 10, 10)
    C:  RMSE, MAE, R^2 = 25.911, 18.120,  0.800
    CV: RMSE, MAE, R^2 = 39.841, 26.266,  0.528
    TST:RMSE, MAE, R^2 = 34.755, 23.038,  0.616
    15.52 seconds

"hidden_layer_sizes":[(10,10,10,10),(20,20,20,20),(40,40,40,40),
    (60,60,60,60),(80,80,80,80),(100,100,100,100)],
    'hidden_layer_sizes': (80, 80, 80, 80)
    C:  RMSE, MAE, R^2 = 23.847, 16.523,  0.831
    CV: RMSE, MAE, R^2 = 35.131, 23.223,  0.633
    TST:RMSE, MAE, R^2 = 35.587, 22.152,  0.597
    90.26 seconds

"hidden_layer_sizes":[(1,1,1,1,1),(2,2,2,2,2),(4,4,4,4,4),
    (6,6,6,6,6),(8,8,8,8,8),(10,10,10,10,10)],
    'hidden_layer_sizes': (10, 10, 10, 10, 10)
    C:  RMSE, MAE, R^2 = 23.676, 17.989,  0.833
    CV: RMSE, MAE, R^2 = 37.943, 25.465,  0.572
    TST:RMSE, MAE, R^2 = 31.726, 21.663,  0.680
    19.01 seconds

"hidden_layer_sizes":[(10,10,10,10,10),(20,20,20,20,20),(40,40,40,40,40),
    (60,60,60,60,60),(80,80,80,80,80),(100,100,100,100,100)],
    'hidden_layer_sizes': (100, 100, 100, 100, 100)
    C:  RMSE, MAE, R^2 = 20.173, 14.513,  0.879
    CV: RMSE, MAE, R^2 = 36.557, 23.692,  0.602
    TST:RMSE, MAE, R^2 = 31.226, 20.542,  0.690
    126.80 seconds

"hidden_layer_sizes":[(1,1,1,1,1,1),(2,2,2,2,2,2),(4,4,4,4,4,4),
    (6,6,6,6,6,6),(8,8,8,8,8,8),(10,10,10,10,10,10)],
    'hidden_layer_sizes': (10, 10, 10, 10, 10, 10)
    C:  RMSE, MAE, R^2 = 27.526, 19.655,  0.775
    CV: RMSE, MAE, R^2 = 38.322, 26.157,  0.563
    TST:RMSE, MAE, R^2 = 36.947, 24.582,  0.566
    16.48 seconds

"hidden_layer_sizes":[(10,10,10,10,10,10),(20,20,20,20,20,20),
    (40,40,40,40,40,40),(60,60,60,60,60,60),(80,80,80,80,80,80),
    (100,100,100,100,100,100)],
    'hidden_layer_sizes': (100, 100, 100, 100, 100, 100)
    C:  RMSE, MAE, R^2 = 19.844, 13.884,  0.883
    CV: RMSE, MAE, R^2 = 34.048, 22.890,  0.655
    TST:RMSE, MAE, R^2 = 31.388, 21.352,  0.687
    137.36 seconds

これだけ探索しても、R^2 > 0.9にも満たない
kNN でさえ、以下の通り、あっさり R^2 > 0.9を達成できるのに
    {'n_neighbors': 1}
    C:  RMSE, MAE, R^2 =  7.916,  3.688,  0.982
    CV: RMSE, MAE, R^2 = 36.334, 22.328,  0.628
    TST:RMSE, MAE, R^2 = 16.748, 11.293,  0.881
    4.23 seconds

データ数が少ないせいか、特徴量が少ないせいかは分からないが、あまり精度が良くない
→
特徴量をhttps://arxiv.org/abs/1803.10260のバージョン(欠損値は飛ばす)に変更
特徴量の数 = 50+1(圧力)
ref:20181122, 1203

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

"hidden_layer_sizes":[(10,10,10),(20,20,20),(40,40,40),
    (60,60,60),(80,80,80),(100,100,100)],
    'hidden_layer_sizes': (40, 40, 40)
    C:  RMSE, MAE, R^2 = 12.544,  8.379,  0.953
    CV: RMSE, MAE, R^2 = 29.972, 17.897,  0.733
    TST:RMSE, MAE, R^2 = 12.745,  9.326,  0.948
    81.61 seconds

"hidden_layer_sizes":[(1,1,1,1),(2,2,2,2),(4,4,4,4),
    (6,6,6,6),(8,8,8,8),(10,10,10,10)],
    'hidden_layer_sizes': (10, 10, 10, 10)
    C:  RMSE, MAE, R^2 = 15.705, 10.622,  0.927
    CV: RMSE, MAE, R^2 = 32.173, 21.041,  0.692
    TST:RMSE, MAE, R^2 = 20.760, 13.577,  0.863
    18.93 seconds

"hidden_layer_sizes":[(10,10,10,10),(20,20,20,20),(40,40,40,40),
    (60,60,60,60),(80,80,80,80),(100,100,100,100)],
    'hidden_layer_sizes': (80, 80, 80, 80)
    C:  RMSE, MAE, R^2 = 12.749,  8.838,  0.952
    CV: RMSE, MAE, R^2 = 27.362, 17.170,  0.777
    TST:RMSE, MAE, R^2 = 15.362, 11.006,  0.925
    114.86 seconds

"hidden_layer_sizes":[(1,1,1,1,1),(2,2,2,2,2),(4,4,4,4,4),
    (6,6,6,6,6),(8,8,8,8,8),(10,10,10,10,10)],
    'hidden_layer_sizes': (10, 10, 10, 10, 10)
    C:  RMSE, MAE, R^2 = 18.687, 12.667,  0.896
    CV: RMSE, MAE, R^2 = 31.948, 21.124,  0.696
    TST:RMSE, MAE, R^2 = 23.554, 16.902,  0.824
    16.91 seconds

"hidden_layer_sizes":[(10,10,10,10,10),(20,20,20,20,20),(40,40,40,40,40),
    (60,60,60,60,60),(80,80,80,80,80),(100,100,100,100,100)],
    'hidden_layer_sizes': (80, 80, 80, 80, 80)
    C:  RMSE, MAE, R^2 = 12.797,  8.777,  0.951
    CV: RMSE, MAE, R^2 = 24.854, 15.826,  0.816
    TST:RMSE, MAE, R^2 = 15.864, 11.310,  0.920
    157.85 seconds

"hidden_layer_sizes":[(1,1,1,1,1,1),(2,2,2,2,2,2),(4,4,4,4,4,4),
    (6,6,6,6,6,6),(8,8,8,8,8,8),(10,10,10,10,10,10)],
    'hidden_layer_sizes': (10, 10, 10, 10, 10, 10)
    C:  RMSE, MAE, R^2 = 17.379, 11.722,  0.910
    CV: RMSE, MAE, R^2 = 27.521, 17.424,  0.775
    TST:RMSE, MAE, R^2 = 22.071, 15.045,  0.845
    16.14 seconds

大幅に性能が向上した！
特に以下の数値。kNNと比べても遜色ない上に、汎化性能が高い
    'hidden_layer_sizes': (80, 80, 80, 80, 80)
    C:  RMSE, MAE, R^2 = 12.797,  8.777,  0.951
    CV: RMSE, MAE, R^2 = 24.854, 15.826,  0.816
    TST:RMSE, MAE, R^2 = 15.864, 11.310,  0.920

Xを51 -> 11(質量を使った変数+圧力)のみ
"hidden_layer_sizes":[(10,10,10,10,10),(20,20,20,20,20),(40,40,40,40,40),
    (60,60,60,60,60),(80,80,80,80,80),(100,100,100,100,100)],
'hidden_layer_sizes': (80, 80, 80, 80, 80)
    C:  RMSE, MAE, R^2 = 24.305, 17.239,  0.824
    CV: RMSE, MAE, R^2 = 39.110, 25.387,  0.545
    TST:RMSE, MAE, R^2 = 29.583, 19.850,  0.722
    128.75 seconds

私の変数の時と大差ない結果に
    'hidden_layer_sizes': (100, 100, 100, 100, 100)
    C:  RMSE, MAE, R^2 = 20.173, 14.513,  0.879
    CV: RMSE, MAE, R^2 = 36.557, 23.692,  0.602
    TST:RMSE, MAE, R^2 = 31.226, 20.542,  0.690
    126.80 seconds

×増やさなくても特徴量を変換すれば良い
○特徴量を増やしたことが良い
ということか？
いずれにせよ、ただディープラーニングにすればいいのではない、ということは分かった

High-Tc候補
formula,P,Tc,AD
GdH9,400,640,1
GdH9,450,649,1
GdH9,500,659,1

GdH10,400,642,1
GdH10,450,652,1
GdH10,500,659,1

HoH9,450,636,1
HoH9,500,646,1

HoH10,450,634,1
HoH10,500,642,1

DyH9,450,627,1
DyH9,500,640,1

DyH10,450,633,1
DyH10,500,641,1


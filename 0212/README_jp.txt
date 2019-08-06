# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 00:15:50 2019

@author: Akitaka
"""

[1] 機械学習
[1a] サイトで勉強　金子研究室
データ解析に関するいろいろな手法・考え方のまとめ
https://datachemeng.com/summarydataanalysis/
学生・研究者へ
https://datachemeng.com/forstudentsresearchers/




[1a2]
主成分分析(Principal Component Analysis, PCA)の前に
変数の標準化(オートスケーリング)をしたほうがよいのか？
https://datachemeng.com/autoscaling_before_pca/
    標準化 = センタリング + スケーリング 
    ただ、わたしの経験の中では、ほとんどスケーリングを行わないほうがよいケースはありません。
    少しでも迷ったら、センタリングに加えてスケーリングも行いましょう。
    
    一応、以下の場合もある
    分散の大きさに比例して主成分に対する変数の重みを大きくしたいときには、スケーリングは行いません。


2018年度「化学プロセスシステム工学」の講義資料を(ほぼ)すべて公開します
https://datachemeng.com/chemical_process_systems_engineering_2018/




[1a3]
https://twitter.com/hirokaneko226/status/1095075178034343936
    Sparse Robust Principal Component Analysis によるプラントの異常検出に関する論文。
    外れ値ほど重みを小さくする共分散行列で robust に、主成分の分散を最大化させるのと一緒に
    ローディングの絶対値を最小化させることでsparseに。Tennessee Eastman (TE) プロセスで検証

https://doi.org/10.1021/acs.iecr.8b04655
"Sparse Robust Principal Component Analysis with Applications
 to Fault Detection and Diagnosis"
Lijia Luo, et al.,
Ind. Eng. Chem. Res. 2019,  58, 3, 1300-1309




https://twitter.com/hirokaneko226/status/1094715302032506881
    高い引張強度をもつポリ乳酸の複合材料の設計に関する論文。論文から135サンプルを収集。
    目的変数を引張強度、説明変数を材料条件や製造条件として決定木で回帰モデル構築。
    モデルによる推定結果や得られた回帰木をみて高い引張強度を達成する材料条件や製造条件を検討

https://doi.org/10.1021/acs.iecr.8b05477
"Implementation of Statistical Learning Methods to Develop Guidelines
 for the Design of PLA-Based Composites with High Tensile Strength Values"
Burak Alakent, et al.,
Ind. Eng. Chem. Res.  XXXX, XXX, XXX-XXX



https://twitter.com/hirokaneko226/status/1094430081131339776
    時系列データにおけるプロセス状態の変化点を検出し、プラントの異常検出を行う論文。
    サンプル間のユークリッド距離や時間幅に基づいて重みを計算し、グラフを作成。
    グラフの変化からプロセス状態の変化を検出。
    Tennessee Eastman processやblast furnace processで検証

https://doi.org/10.1021/acs.iecr.8b02455
"Unsupervised Change Point Detection Using a Weight Graph Method
 for Process Monitoring"
Ruqiao An, et al.,
Ind. Eng. Chem. Res. 2019,  58, 4, 1624-1634




https://twitter.com/hirokaneko226/status/1093758554324951040
    第９章 プラントの設計、運転、設備管理とスマート化の、
    6節 ソフトセンサーを活用したプラントのスマート化事例、について執筆しました。

http://www.gijutu.co.jp/doc/b_1985.htm#9
化学プロセスのスケールアップ、連続化



[1b] 遺伝的アルゴリズム
[1b1] プログラミング
※vclreaxはしない
0208
大体できたので、繰り返してみるテスト
途中で、収束しないＮＧばかリに！
=>
sortが逆順だった！直す

vcrelaxに直す
=>
vcrelax終了後、fit_geneを計算するのは出来た

0212
vcrelaxでテスト C2つでダイヤ出るか？
[!]
    αβγをすべてランダムにしていたが、
    α>β+γのような場合、平行六面体が成立しない！
    そのために、エラーが頻発しているのではないか
=>
αβγで、0 =< cos < 0.9とした

[!] b/a, c/a = [0:1]しかとれない！
=>
b,cもaと同様の発生をさせて、aで割ることにした




[1c] 遺伝的プログラミング
[todo]
    遺伝的プログラミングによる特徴量生成
    https://qiita.com/overlap/items/e7f1077ef8239f454602
    ->Tc計算式そのものもGAでやる(石河さんと同じこと)

[1c1]
https://qiita.com/overlap/items/e7f1077ef8239f454602
のプログラムを、分類→回帰に変更したものをまず作成

少しエラー
eval_genfeatで返すのはタプルなので、「return 変数」ではなく「return 変数,」であるべき
ref:
https://github.com/DEAP/deap/issues/256

完成！
OLS, 評価 MAE, 「交叉確率50%、突然変異確率10%、10世代まで進化」×100まで計算
[!] MAEは小さいほどいい。サンプルのAUCは大きいほどいい。逆なので、マイナスをかけた状態
### Results
Baseline MAE train : -1.993516463016931e-13
Baseline MAE test : -2.673417043297377e-13
Best MAE train : -9.083844787483031e-14
Best MAE test : -3.0127011996228247e-13

ベースラインよりMAE_train、test両方とも向上。成功。

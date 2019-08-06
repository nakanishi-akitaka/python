# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:43:15 2019

@author: Akitaka
"""

[1] 機械学習
[1a] サイトで勉強　金子研究室
データ解析に関するいろいろな手法・考え方のまとめ
https://datachemeng.com/summarydataanalysis/
学生・研究者へ
https://datachemeng.com/forstudentsresearchers/


[1a2]
https://twitter.com/hirokaneko226/status/1095804496221954048
    ADMET(吸収、分布、排泄、代謝、毒性)を推定する
    マルチタスクディープニューラルネットワークに関する論文。
    ChEMBLや社内で収集した約五万化合物を利用。
    記述子はDRAGON, MOE, RDKitで計算。
    ニューラルネットワークによる推定値と化学構造の類似性で部分構造の寄与を検討

https://doi.org/10.1021/acs.jcim.8b00785
"Predictive Multitask Deep Neural Network Models for ADME-Tox Properties:
 Learning from Large Data Sets"
Jan Wenzel, et al.,
J. Chem. Inf. Model.  XXXX, XXX, XXX-XXX


[1a3] 人工知能の自然言語処理
東ロボくんの生みの親《新井紀子》教授の間違いが明らかになった日。
人間は人工知能に読解力でも負けつつある
https://togetter.com/li/1285134

https://twitter.com/_Ryobot/status/1050925881894400000
GLUEベンチマークの全言語理解タスクでぶっちぎりのSOTA！ 質疑応答タスクのSQuADでもSOTA！ 
話題の手法BERTってなにっ？（24層16注意ヘッドのTransformer！？）

https://arxiv.org/abs/1810.04805
"BERT: Pre-training of Deep Bidirectional Transformers
 for Language Understanding"
Jacob Devlin


(2019年1月26日更新)
自然言語処理の11個のタスクで最高精度へと導いた「BERT」のその後についてまとめた
https://togetter.com/li/1299751

https://twitter.com/jaguring1/status/1069787936474841088
良い記事
「自分たちで学習した BERT が有用であることがひと目で理解できます」 
「fine-tuning なので少量のデータで良い結果を出せるところが強力」 「応用上かなり有用」
=>
BERT with SentencePiece で日本語専用の pre-trained モデルを学習し、それを基にタスクを解く
https://techlife.cookpad.com/entry/2018/12/04/093000
まとめ
    BERT の multilingual モデルは日本語の扱いには適さないので SentencePiece を使った
        tokenization に置き換えて学習
    pre-training にはクックパッドの調理手順のテキスト（約1600万文）を使用
    学習は p3.2xlarge インスタンスで 3.5 日程度学習を回し、
        loss は以下の図のように推移（バッチサイズ32）
    学習済みの pre-trained モデルを基に、
        手順のテキストが料理の手順であるか否かを予測する問題を解き、以下の表のように良い結果
    （ある程度ドメインを限定すれば）現実的なコストで有用な pre-trained モデルが作れる



[1b] 遺伝的アルゴリズム
[1b1] プログラミング
0212 vcrelaxでテスト C2つでダイヤ出るか？第0-5世代
0213 第5-13世代
0214 一旦終わるためのプログラム作成 step3.sh
    選択でのsortが逆だった！
        ただ、エラーが出たとき用のE=1000000としたのが、一番上に来てしまう
            100000, -50, -40, -30,...となる。プラスマイナスは無視？
        従って、E>10000をはじくようにした
    arccos の |α+β|がπ*5/6を超えないように修正
    =>
    C4つで第0世代から計算し直し＋スクリプトアップデート
    現在第0-1世代

[?] そもそも、最初を完全にランダムにする必要があるのか？
    格子定数や原子位置の組み合わせとして、有名なもの、対称性の良いものも
    ランダムなものと混ぜたらいいのでは？
[?][!] 同じ構造の固体が出来てしまった場合、選別した者が同じものばかりになってしまう！
    どのように解決する？手動変更というゴリ押しもあるが



[1c] 遺伝的プログラミング
[todo]
    遺伝的プログラミングによる特徴量生成
    https://qiita.com/overlap/items/e7f1077ef8239f454602
    ->Tc計算式そのものもGAでやる(石河さんと同じこと)

[1c1]
0213
Tcに応用
OLS, 評価 MAE, 「交叉確率50%、突然変異確率10%、10世代まで進化」×10
0214
=>100まで増やして再計算

ベースラインよりMAE_train、test両方とも向上。
10回のに比べると、trainのみ向上。やや過学習？
### Results
Baseline MAE train : -34.31809188901318
Baseline MAE test : -40.14682923813623
Best MAE train : -12.139364801048485
Best MAE test : -33.2555437874312


ref
    20190213
    ### Results
    Baseline MAE train : -34.31809188901318
    Baseline MAE test : -40.14682923813623
    Best MAE train : -17.36988941667709
    Best MAE test : -24.475161698980138


[1c2] KNNやSVRなどでも試す
    ハイパーパラメータは普通にTcを学習した時のハイパーパラメータで固定してしまうかデフォルト

kNN k=5(デフォルト)
### Results
Baseline MAE train : -35.736930467132865
Baseline MAE test : -44.68545707317073
Best MAE train : -19.5332052
Best MAE test : -38.58145780487805
意外にも、ベストでさえOLSよりスコアが低い＋特徴量作成無しの方がマシ

ref:特徴量作成なしで、普通にCV＋kNN
Best parameters set found on development set:
{'n_neighbors': 1}
C:  RMSE, MAE, R^2 =  7.249,  3.509,  0.983
CV: RMSE, MAE, R^2 = 34.853, 18.989,  0.610
TST:RMSE, MAE, R^2 = 17.007, 11.554,  0.930


kNN k=1(CV結果より)
### Results
Baseline MAE train : -25.66501008857809
Baseline MAE test : -46.67845243902438
Best MAE train : -12.30930787878788
Best MAE test : -27.376343902439025
それでも今一つ
=>
スケーリング忘れてた！
入れ直して再計算
しかし、余り改善されず？


kNN k=1(CV結果より)
### Results
Baseline MAE train : -17.815157519813518
Baseline MAE test : -30.699521951219506
Best MAE train : -13.361390340326341
Best MAE test : -30.000828048780487

=>
cv = 5から
cv = KFold(n_splits=3, shuffle=True) に合わせる
kNN k=1(CV結果より)
### Results
Baseline MAE train : -17.73453293408087
Baseline MAE test : -38.19279268292683
Best MAE train : -13.085091086193225
Best MAE test : -29.553540243902432

=>
cv = KFold(n_splits=3->5    にする
理由
    今後、GridSearchCVのデフォルトが3=>5になることを受けた
    train_test_splitでtest_size=0.2としたことから

### Results
Baseline MAE train : -17.19945611188811
Baseline MAE test : -31.39276463414634
Best MAE train : -13.946592787878789
Best MAE test : -33.73143536585365


=>
SVR(C=256, epsilon=1.0, gamma=2) (CVの結果より)
こちらでもあまり改善はされない

### Results
Baseline MAE train : -22.294466903919638
Baseline MAE test : -28.56859164070416
Best MAE train : -13.939239552652754
Best MAE test : -22.855579378847423

### Generated feature expression
tan(tan(tan(add(tan(ARG1), ARG1))))
sin(ARG2)
cos(sub(ARG3, sub(ARG3, ARG3)))
tan(mul(ARG5, ARG7))

=>
kNNに戻して、スコアをR2に変える
かなりの過学習では？
前のもそれなりに傾向はあったが

### Results
Baseline SCORE train : 0.734361489882094
Baseline SCORE test : 0.02347146623074714
Best SCORE train : 0.8593634487330952
Best SCORE test : 0.1730651955788547

=>
OLSに戻す
### Results
Baseline score train : 0.3238581621358453
Baseline score test : -0.10605714216942785
Best score train : 0.7436476331878946
Best score test : 0.13625870199501655


=>
計算方法の問題！
    y_incv = cross_val_predict(rgr, X_test, y_test, cv=cv)
    cross_val_predictだと、また{X,y}_testで学習し直すのだが、データ数が減り過ぎるのが問題
    =>
    y_incv = rgr.predict(X_test)
としたら
kNN k=1(CV結果より) + スケーリング + KFold(n_splits=5, shuffle=True)
### Results
Baseline score train : 0.5564514858923418
Baseline score test : 0.8847188618882776
Best score train : 0.8797080522068722
Best score test : 0.9230270859229217


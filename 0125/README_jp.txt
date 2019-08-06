# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:14:15 2019

@author: Akitaka
"""
[1] 機械学習
[1a] サイトで勉強　金子研究室
データ解析に関するいろいろな手法・考え方のまとめ
https://datachemeng.com/summarydataanalysis/


[1a2]
https://twitter.com/hirokaneko226/status/1088558970455830528
化学構造の官能基をファーマコフォアの特徴を表す超原子で置き換えた reduced graph から、 
SMILES に変換することで化学構造を生成する論文。
reduced graph と SMILES との関係をディープニューラルネットワークで構築。
Reduced graph をそれに入力して SMILES を生成

https://doi.org/10.1021/acs.jcim.8b00626
"De Novo Molecule Design by Translating from Reduced Graphs to SMILES"
Peter Pogány, et al.,
J. Chem. Inf. Model.  XXXX, XXX, XXX-XXX

[word] 
ファーマコフォア（英: pharmacophore）は、生体高分子によるリガンドの分子認識に必要な
    分子の特徴（官能基群とそれらの相対的な立体配置）の（抽象的な）概念である。
ファーマコフォアとは、医薬品のターゲットとの相互作用に必要な特徴を持つ官能基群と、
    それらの相対的な立体配置も含めた（抽象的な）概念である。
超原子 (英語: superatom) は、複数の原子が相互作用により凝縮したクラスター（塊）である。
SMILES記法（スマイルスきほう、英: Simplified molecular input line entry system）とは、
    分子の化学構造をASCII符号の英数字で文字列化した構造の曖昧性の無い表記方法である。

[1a3] 単語検索
[word]
正則化（せいそくか、英: regularization）とは、
    数学・統計学において、特に機械学習と逆問題でよく使われるが、
    機械学習で過学習を防いだり、逆問題での不良設定問題を解くために、
    追加の項を導入する手法である。
    正則化の理論的正当化はオッカムの剃刀にある。
    ベイジアンの観点では、多くの正則化の手法は、モデルのパラメータの事前情報にあたる。 
    誤差関数(二乗誤差など) → 誤差関数 + 正則化項 λ*Σ_i|w_i|**p
    ここで、p=1のときの正則化項をL1ノルム、それによる正則化をL1正則化という(1→2も同様)
    ref: 
    wikipedia
    http://breakbee.hatenablog.jp/entry/2015/03/08/041411
良設定問題/不良～
    以下の3つの条件を満たす/満たさない問題
    解が存在する
    解が一意である
    パラメタを連続的に変化したとき、解も連続的に変化する
オッカムの剃刀
    「ある事柄を説明するためには、必要以上に多くを仮定するべきでない」とする「指針」
        説明に不必要であることは、存在の否定ではない
        真偽の判定則ではない
赤池情報量規準（ Akaike's Information Criterion:AIC）
    統計モデルの良さを評価するための指標である。
    「モデルの複雑さと、データとの適合度とのバランスを取る」ために使用される。
    AIC最小のものを選択すれば常に最良であるかと言うと一概にはそう言えない。

[1a4] 外れ値検出, kNN とLOF
異常検知ビジネスで活用できる外れ値検知手法まとめ
http://hktech.hatenablog.com/entry/2018/10/24/000244
Local Outlier Factor (LOF) による外れ値検知についてまとめた
http://hktech.hatenablog.com/entry/2018/09/04/002034
PythonでLocal Outlier Factor (LOF)を実装してみた
http://hktech.hatenablog.com/entry/2018/10/10/232427

以上、3つの記事によると、
    kNN:近傍のk個の点との距離の最大値がある閾値を超えれば、外れ値
    LOF:近傍のk個の点との距離の平均の逆数 = 局所密度と、
        近傍のk個の点の局所密度の差が、ある閾値を超えれば、外れ値
私が実装したkNNによるADの計算方法とはどちらも異なる。LOFの方が近い事は近いが。
LOFはscikit-learnにライブラリー有
https://scikit-learn.org/stable/modules/generated/
    sklearn.neighbors.LocalOutlierFactor.html


[1a7] outlierとnoveltyの違い
https://scikit-learn.org/stable/modules/outlier_detection.html
    outlier:学習データに含まれる、異常なデータ
    novelty:学習データとは異なる新しいデータに含まれる、異常なデータ
        LOFはデフォルトでは、outlier検出のみ .fit_predictを使う
        novelty=Trueとすることで、novety検出が可能。.predictを使う

https://scikit-learn.org/stable/auto_examples/neighbors/
    plot_lof_novelty_detection.html
LOFによる"novelty" detectionのサンプル
novelty = True が原因でエラー！

以前にもLOFは使ったことあるが、そのときはnovelty=Trueは使わず
20180621/test4_plot_outlier_detection.py
20180713/test2_LOF.py
20180713/test3_OCSVM_EE_IF_LOF.py


spyderをアップデート
https://pypi.org/project/spyder/
https://anaconda.org/anaconda/spyder

改めて実行してみたけれど、ダメだった
原因不明だが、LOFでなければダメな理由も特にないので放置する。


[1a6] 他のoutlier検出方法
https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html
    scikit-learnのoutlier検出方法4つで比較
        OneClassSVM:説明略
        IsolationForest:ランダムフォレストの応用
        LocalOutlierFactor:先述の通り
        EllipticEnvelope:データの中心を中心とした1つの楕円で識別する


https://scikit-learn.org/0.18/modules/outlier_detection.html
    2.7.2.3. One-class SVM versus Elliptic Envelope versus Isolation Forest
    LOFを除いた、3つで比較
    厳密には、OCSVMはoutlier検出ではなく、novelty検出。
    しかし、高次元だったり、データ分布に対する前提が不明な場合では、
    outlier検出が難しく、OCSVMが有益な情報をもたらす。
    
    1.inlierの分布が楕円形：EnvelopeとForestはOK。OCSVMがやや過剰適合でNG
    2.inlierの分布が2つの塊：EnvelopeとOCSVMがNG。
    3.inlierの分布がガウス分布でない：EnvelopeがNG。ForestはOK。OCSVMは妥当な近似。
        これだけ見ると、Forestが良い？と思うが


[1a7] isolation forest
不動産価格を機械学習で予測するKaggleに挑戦する [発展編1:上位20%]
http://ochearno.net/house_prices_stacking-2
    Isolation Forestは名前が示す通り決定木ベースの外れ値検知法で、
    特徴量をランダムに選び、決定境界を適当に何度も引きます。
    その後、あるサンプルを集団から分離させるのに必要だった分割回数(線を引いた回数)
    を利用して、外れ値を検出します。
    外れ値ほど分離しやすく分割回数は減り、
    データが密集している付近の値ほど分離しにくく分割回数が増えるという具合です。

    決定木ベースなので、ランダムフォレストなどと同じく正規化等なくてもうまい具合に動いてくれる上、
    データが多次元正規分布に従っているなどの仮定も必要ないのでとても便利に扱えます。
    ツリーベースのアルゴリズムはこの安定・安心感が素晴らしいと思います。

余談ながら、上の記事より
[word] スタッキング
    更に、Kaggleで流行っているスタッキングという手法を使いたいと思います。
    スタッキングはアンサンブル学習の一種とも考えられ、複数の学習器を組み合わせて予測を行うことです。
    具体的には予測を段階にわけて、第一段階は各種機械学習のアルゴリズムを出力し、
    第二段階の入力としてその出力を別の機械学習アルゴリズムに流し込んで予測するという具合です。
    第二段階は機械学習アルゴリズムを通しても良いし、
    単に平均、多数決、重み付けなどでも問題ないようです。
    
    スタッキングで精度が上がる理由としては、それぞれの学習器とチューニングによる
    バイアスとバリアンスがうまいことならされる・・・
    ことかなと考えますが、アンサンブル学習の弱学習器が強い奴になった感じというざっくり理解で使います。



[1a8] 更に別の検出方法
DLLab 異常検知ナイト 資料 20180214 
https://www.slideshare.net/KosukeNakago/dllab-20180214-88470902
    Kernel Density Estimation (KDE)
    Gaussian Mixture Model (GMM) 
    Local Outlier Factor (LOF) 
    Isolation Forest (IF) 
    One Class SVM （OCSVM）
    この5つでテスト計算して比較している
    それらの用途や長所短所のまとめがp.25。

Deep Learning Lab初のエンジニア向けイベント「異常検知ナイト」レポート
https://thinkit.co.jp/article/14039
    上記スライドが使われたイベントのレポートなので、同じ内容
    スライド以外にも説明が入るので、そこが知りたければ






# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 17:17:59 2019

@author: Akitaka
"""
20180107
[1a2] 新着記事
データ解析・機械学習をはじめたいとき、市販のソフトウェアを使うのがよいか、
プログラミングを勉強するのがよいか、それぞれのメリット・デメリットを考える 
https://datachemeng.com/software_programming_merit_demerit/


2018年度データ化学工学研究室(金子研究室)成果報告会をやります！ 
https://datachemeng.com/meetingonlinesalon2018/

[1a3] 金子研究室twitterより
https://twitter.com/hirokaneko226/status/1078757934203428864
    いろいろな recurrent neural network (RNN) で12の時系列データを用いて
    予測性能を比較した論文。
    Elman, Jordan, fully connected RNN, NARMAX, LSTM で検討。
    back propagation through timeやサポートベクター回帰で予測誤差が小さいこともあったが
    RNNで計算時間短縮できた
->
https://doi.org/10.1016/j.neucom.2018.09.012
    "On extreme learning machines in sequential and time series prediction:
     A non-iterative and approximate training algorithm
     for recurrent neural networks"
    Yara Rizk, Mariette Awad,
    Neurocomputing Volume 325, 24 January 2019, Pages 1-19

[word] RNN (Recurrent Neural Network)
https://qiita.com/kiminaka/items/87afd4a433dc655d8cfd
    RNNの利点は文章など連続的な情報を利用できる点です。
    従来のニューラルネットワークの考え方はそうではなく、
    インプットデータ（またアウトプットデータも）は互いに独立の状態にある、と仮定します。
    しかしこの仮定は多くの場合、適切ではありません。
    例えば次の言葉を予測したい場合、その前の言葉が何だったのかを知っておくべきですよね？
    RNNのRはReccurent（再帰）という意味で、直前の計算に左右されずに、
    連続的な要素ごとに同じ作業を行わせることができます。
    言い方を変えると、RNNは以前に計算された情報を覚えるための記憶力を持っています。
    理論的にはRNNはとても長い文章の情報を利用することが可能です。



https://twitter.com/hirokaneko226/status/1078807024111112193
    F0F1-ATPase阻害剤としてのカーボンナノチューブ(CNT)のナノ毒性を判別する論文。
    CNTのSEM画像から計算した記述子を入力変数にして、
    線形判別分析やニューラルネットワークで暮らす分類モデルを構築。
    バリデーションデータでspecificityとsensitivityが99%以上を達成
->
https://doi.org/10.1021/acs.jcim.8b00631
    "MitoTarget Modeling Using ANN-Classification Models
     Based on Fractal SEM Nano-Descriptors: Carbon Nanotubes
     as Mitochondrial F0F1-ATPase Inhibitors"
    Michael González-Durruthy et al.,
    J. Chem. Inf. Model.  XXXX, XXX, XXX-XXX


https://twitter.com/hirokaneko226/status/1079183149165248512
    半教師あり学習によるディープニューラルネットワークでクラスタリングする論文。
    KLダイバージェンスと一緒に、アサインされたクラスターを考慮したサンプル間の制約も最小化する。
    後者において教師ありデータを考慮する。５つの画像データやテキストデータで検証
->
https://doi.org/10.1016/j.neucom.2018.10.016
    "Semi-supervised deep embedded clustering"
    Yazhou Ren et al.,
    Neurocomputing Volume 325, 24 January 2019, Pages 121-130


https://twitter.com/hirokaneko226/status/1080655616190832640
    タンパク質間相互作用を0,1で予測するディープニューラルネットワークに関する論文。
    記述子をアミノ酸配列に基づく３種の記述子と、ニューラルネットワークの構成9通りを変え、
    アンサンブル学習。5-fold クロスバリデーションで７つの指標でRF, SVM, kNNと比較検討
->
https://doi.org/10.1016/j.neucom.2018.02.097
    "Protein–protein interactions prediction based on
     ensemble deep neural networks"
    LongZhang et al., 
    Neurocomputing Volume 324, 9 January 2019, Pages 10-19

https://twitter.com/hirokaneko226/status/1080665555286122503
->
https://t.co/fkKv0Uynxj
    "Helping institutions understand how researchers are sharing their data"
    Iain Hrynaszkiewicz

https://twitter.com/hardmaru/status/1075944352214441984
    This paper: “We train a classifier to predict 
    whether a paper should be rejected based solely on 
    the visual appearance of the paper. 
    Results show that our classifier can safely reject 50% of the bad papers 
    while wrongly reject only 0.4% of good papers.”
->
https://arxiv.org/abs/1812.08775
    "Deep Paper Gestalt"
    Jia-Bin Huang

https://twitter.com/hirokaneko226/status/1080981788946644993
    ディープニューラルネットワーク・ランダムフォレスト・重み付きk最近傍法を用いて、
    21のデータセットで活性や毒性の予測性能を比較した論文。
    ３つの手法間の性能差よりも、トレーニングデータとの距離、つまりモデルの適用範囲、
    によって推定性能が変化する影響が大きかった
->
https://doi.org/10.1021/acs.jcim.8b00348
    "Dissecting Machine-Learning Prediction of Molecular Activity:
     Is an Applicability Domain Needed
     for Quantitative Structure–Activity Relationship Models 
     Based on Deep Neural Networks?"
    Ruifeng Liu et al.,
    J. Chem. Inf. Model.  XXXX, XXX, XXX-XXX
    # > ３つの手法間の性能差よりも、トレーニングデータとの距離、つまりモデルの適用範囲、
    # > によって推定性能が変化する影響が大きかった
    # アブストラクトにある図を見れば分かる。
    # 「機械学習は補完に過ぎない」という考えが正しいのならば、こうなるのも納得は行く。
    #     補完の種類がどうであれ、データの間をつなぐ以上は大した差が出ない。
    #     データ同士の距離が遠い(データ密度が小さい)かどうかの方が重要。
    # 考えの証拠としては弱いけれど。


https://twitter.com/hirokaneko226/status/1081343308746522624
    6つのanticancer targetの阻害剤として50万化合物用いて学習させた
    ディープニューラルネットワークに関する論文。ランダムフォレストと比較して良好な性能。
    PubChemの2015年以降の阻害剤の50％を0.01〜0.09％の偽陽性率で同定。
        コードはこちら 
        https://github.com/xiaotaw/chembl 
        # パッと見たところ、頑張れば読み切れそうな長さ。
->
https://doi.org/10.1002/minf.201800031
    "Development of Ligand‐based Big Data Deep Neural Network Models
     for Virtual Screening of Large Compound Libraries"
    Tao Xiao, et al.,

https://twitter.com/JCIM_ACS/status/1081535822267105283
    Sparse Generative Topographic Mapping 
    for Both Data #Visualization and #Clustering
->
https://doi.org/10.1021/acs.jcim.8b00528
    "Sparse Generative Topographic Mapping for
     Both Data Visualization and Clustering"
    Hiromasa Kaneko
    J. Chem. Inf. Model. 2018,  58, 12, 2528-2535
        以下のページで、金子先生本人が解説してる
        https://datachemeng.com/sgtm/
->
すでに、読んだことのあるページ
20181029<div>
[1a2] 新着記事
Sparse Generative Topographic Mapping(SGTM):
データの可視化とクラスタリングを一緒に実行する方法 [金子研論文]
https://datachemeng.com/sgtm/
https://doi.org/10.1021/acs.jcim.8b00528
比較１
GTM：
    データの可視化手法、SOMの上位互換
SGTM:
    GTM のアルゴリズムを少し改良して、モデルを sparse にすることで、
    データの可視化だけでなくクラスタリングも一緒にできるようになった
比較２
GTM：
    混合係数(負担率)πk=1/(GTMマップのグリッド数)=1/(マップサイズ)**2、
    共分散を0,分散をある値に固定した GMM
    GMMを二次元に落とし込んだもの。
GMM:
    クラスタリング手法。混合係数 πk が可変。
SGTM:
    πkを可変にして、W, β と一緒に Expectation-Maximization(EM)アルゴリズムで
    πk を最適化する手法
    ベイズ情報量基準(Bayesian Information Criterion, BIC)を用いて、
    クラスター数が自動できまる。

テスト：
    QSPR のデータセットや QSAR のデータセットを解析
結果：
    データの可視化の性能を表す指標 k3n error の値もほとんど変わらない
    GTM と同様に可視化ができ、さらにクラスタリングも可能であることを確認
    各サンプルに自動的にクラスターが割り当てられ、
    色付きのサンプルとして二次元にプロットされて、とても見やすい

サンプルコード
https://github.com/hkaneko1985/gtm-generativetopographicmapping
</div>


https://twitter.com/hirokaneko226/status/1081680830974484480
    PI3Kα と tankyrase との dual target 阻害剤のための、
    ドッキングシミュレーションとクラス分類との論文。
    AutoDock Vina でドッキングして得られたリガンドの三次元構造から記述子を計算。
    SVM, kNN, RF, LDA, DNN でクラス分類モデルを構築。バーチャルスクリーニングへ
->
https://doi.org/10.1002/minf.201800030
    "Machine Learning Classification Models
     to Improve the Docking‐based Screening:
     A Case of PI3K‐Tankyrase Inhibitors"
    Vladimir P. Berishvili, et al.,

# こうして一気読みして思うのは、kNNの意外な採用率。
# RFやSVRの採用率が高いのは分かるけど。
# 比較対象という名の当て馬かもしれないが、それであっても、
# 「kNNに勝てたって大したことない」とは思われてない、からこそ比較に出されているのかもしれない。





20180106
[1b] 獲得関数は相互情報量(Mutual Information)がオススメ？
https://tkng.org/b/2015/07/31/bayesian-optimization/
    Mutual Informationを使っておくのがいいそうである。

http://d.hatena.ne.jp/jetbead/20150720/1437364316
    先週のNL研のニコ生で、ベイズ的最適化についての招待講演を見ていて
    「SEは滑らかすぎる」という発言がよくわからなかったので、GP-MIを試してみる

https://www.slideshare.net/issei_sato/bayesian-optimization
    実用的にもよい

https://pythonoum.wordpress.com/2018/09/09/parameter-tuning/
    Mutual Information[Contal+2014]がよく使われます。

https://qiita.com/voyager/items/7b2d461c4a0b3fdc4309
    獲得関数にUCBを使うよりもmutual informationが使うほうがよいと聞いた。

https://nykergoto.hatenablog.jp/category/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92
    敵対的サンプルを判別する基準として相互情報量 (Mutual Information) 
    が優れていることを主張する論文.






20180105
[1] 機械学習
[1a] サイトで勉強　金子研究室
データ解析に関するいろいろな手法・考え方のまとめ
https://datachemeng.com/summarydataanalysis/
一週間の予定
月：数学(行列計算・線形代数・統計・確率)が不安な方へ, データの前処理
火：データセットの可視化・見える化, クラスタリング, 変数選択
水：回帰分析
木：クラス分類, アンサンブル学習, 半教師あり学習 (半教師付き学習)
金：モデルの検証, モデルの適用範囲, モデルの解釈, モデルの逆解析
土：実験計画法, 時系列データ解析 (ソフトセンサーなど)
日：異常検出・異常診断, 失敗例, その他




[1b] ベイズ最適化テストプログラム
[1b1] テストプログラムの仕様変更
https://github.com/hkaneko1985/design_of_experiments/tree/master/Python
    design_of_experiments.py
https://qiita.com/masasora/items/cc2f10cb79f8c0a6bbaa
https://github.com/Ma-sa-ue/practice/blob/master/machine%20learning(python)/
    bayeisan_optimization.ipynb
を基にして、他の獲得関数を実装する
UCB + PI, EI, MIは実装完了

最適化するべき関数を代入する形式にする
一々新しいファイルを作るのは無駄
    blackbox_func = -Rosenbrock_func


[1b2] ベイズ最適化のサンプル 
[?] ベイズ最適化といっても、モデル＋グリッドサーチによる最適化(逆解析)にちょっと変化付けただけ？
獲得関数 = UCBでk=0とすれば、そうなる
[?] k=0を試すとどう変化する？
→以下の通り、2パターンのUCBで試す。


    獲得関数 PI, EI, UCB(k=10), UCB(k=0)
    探索範囲 [x1, x2] = [-10,-10],...,[10,10], step = 0.2
    試行回数 n_iteration = 100

Ackley function 
    y = 20 - 20*exp(-0.2*sqrt((x1**2+x2**2)*0.5)
         + e - exp((cos(2*pi*x1)+cos(2*pi*x2))*0.5)
    厳密解は x = (0,0), y = 0
    ACQ, iter, x_best, y_best 
    PI   93 [[0. 0.]] [-0.]
    EI   59 [[0. 0.]] [-0.] 
    UCB  72 [[0. 0.]] [-0.]
    UCB0  1 [[0. 0.]] [-0.]
    MI   60 [[0. 0.]] [-0.]

Sphere function 
    y = x1**2 + x2**2
    厳密解は x = (0,0), y = 0
    ACQ, iter, x_best, y_best 
    PI  100 [[-0.8 -8.8]] [-78.08]
    EI   80 [[0. 0.]] [-0.]
    UCB  50 [[0. 0.]] [-0.]
    UCB0  1 [[0. 0.]] [-0.]
    MI  100 [[-0.2  0. ]] [-0.04]

Rosenbrock function
    y = Sum_i=1^n-1 (100*(x_i+1 - x_i**2)**2 + (x_i - 1)**2)
    厳密解 x = (1,1), y = 0
    ACQ, iter, x_best, y_best 
    PI  100 [[-0.4 -6.8]] [-4846.12]
    EI  100 [[ 3.2 10. ]] [-10.6]
    UCB 100 [[0.4 0.4]] [-6.12] 
    UCB0 11 [[0. 0.]] [-0.]
    MI  100 [[-1.8  2.4]] [-78.4]
    やたら成績が悪いのはなぜ？　一番シンプルなUCB(k=0)のみが正解

Beale function
    y = (1.5 - x1 + x1*x2)**2
     +  (2.25 - x1 + x1*x2**2)**2
     +  (2.625- x1 + x1*x2**3)**2
    厳密解 x = (3,0.5), y = 0
    ※ xのステップ幅から、0.5は計算できない。0.4, 0.6まで 
    ACQ, iter, x_best, y_best 
    PI   100 [[ -0.8 -10. ]] [-651519.123125]
    EI   100 [[2. 0.]] [-0.703125]
    UCB  100 [[3.2 0.6]] [-0.10270644] 
    UCB0 100 [[1.2 0. ]] [-3.223125]
    MI   100 [2.4 0.2]] [-0.23894964]
    PIがダメダメ

Booth function
    y = (x1 + 2*x2 -7) + (2*x1 + x2 - 5)
    厳密解 x = (1,3), y = 0
    ※ xのステップ幅から、0.5は計算できない。0.4, 0.6まで 
    ACQ, iter, x_best, y_best 
    PI  100 [[ 7.6 -5.6]] [-133.52]
    EI   34 [[1. 3.]] [-0.]
    UCB  48 [[1. 3.]] [-0.]
    UCB0 21 [[1. 3.]] [-0.]
    MI  100 [[1.  2.6]] [-0.8]
    PIがダメダメ

Matyas function
    y = 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2
    厳密解 x = (0,0), y = 0
    ※ xのステップ幅から、0.5は計算できない。0.4, 0.6まで 
    ACQ, iter, x_best, y_best 
    PI   64 [[0. 0.]] [-0.]
    EI   82 [[0. 0.]] [-0.] 
    UCB 100 [[-0.6 -0.8]] [-0.0296]
    UCB0  1 [[0. 0.]] [-0.]
    MI  100 [[-0.2  0. ]] [-0.0104]

Levi function N.13
    y = sin(3*pi*x1)**2 
        + (x1-1)**2 * (1+sin(3*pi*x2))
        + (x2-1)**2 * (1+sin(2*pi*x2))
    厳密解 x = (0,0), y = 0
    ※ xのステップ幅から、0.5は計算できない。0.4, 0.6まで 
    ACQ, iter, x_best, y_best 
    PI   100 [[2.4 1.2]] [0.41381376]
    EI    38 [[ 1.8 -0.2]] [0.84925407] なぜかここで収束扱い
    UCB  100 [[0.6 1.2]] [0.50191203]
    UCB0 100 [[1.8 1.8]] [0.88840886]
    MI   100 [[-5.6  1.2]] [-1.62223516]

MI, UCBが高速。
MIは収束まではが速い。収束値が少しずれている。




[1b3] 獲得関数についての疑問
https://www.slideshare.net/hoxo_m/ss-77421091
    △ UCBは最適値へ収束する理論的保証あり
    ◯ 正確には、「累積Regret R_T が Regret上限 G(T) √α 以下になる確率」が大きい
        Pr[R_T ≦ G(T) √α] ≧ 1-δ
        α = log(2/δ)

とあったものの、これってMIのことではないか？という疑問が出てきた
    αとかδとか、UCBには出てこないパラメータ
    MIの論文を読んだら、Pr[R_T ≦ ...] ≧ 1-δの形の式がでてきた。...の部分は違うが。
    少なくとも、「累積Regretの上限はUCBよりも低いオーダーになる」ということは論文から分かった
ref:
Gaussian Process Optimization with Mutual Information
https://arxiv.org/abs/1311.4825v3































20180104
[1a2] 混合ガウスモデルの考察
[Pythonコードあり] 教師あり混合ガウスモデル(Supervised Gaussian Mixture Models)で
回帰分析も逆解析も自由自在に♪～Gaussian Mixture Regression(GMR)～
https://datachemeng.com/gaussianmixtureregression/
ガウス過程回帰と比べて
    1.カーネル法は使っていない
    2.複数の正規分布を使用する

https://github.com/hkaneko1985/sgmm
https://github.com/hkaneko1985/sgmm/blob/master/demo_gmr.py
https://github.com/hkaneko1985/sgmm/blob/master/gmr.py
デモンストレーションでは、GMMRの出力は、以下の4つ
    重みが最大である正規分布の値(mode)
    正規分布の重みつき平均値(mean)
    各正規分布の値
    各正規分布の重み

ガウス過程回帰の比較、ベイズ最適化への応用について
こちらの場合、yの推定値は、
    1.重みが最大である正規分布の値 (mode)
    2.複数の正規分布の重みつき平均値 (mean)
故に、平均値μ & 標準偏差σ、を得ようと思ったら、少し工夫が必要そう
    単純に考えれば、1.から正規分布の数を1つにすれば、その正規分布のμ,σを使えばいい
しかし、無理矢理ベイズ最適化できるようにすることもないのでは？
    自前で逆解析ができる
    正規分布の数がハイパーパラメータである

[?] ベイズ最適化中はハイパーパラメータの変更がそもそも必要なのか？
テスト計算では、初期サンプルが2点のみ、その後の試行回数が50回とアンバランス。
    試行コストが高いからこそ、ベイズ最適化でターゲットを絞ってやるハズ
これだけ試行回数できるのなら、最初からある程度データとってから、
ハイパーパラメータの最適化をした上でモデルを構築できるのでは？
それならば、ガウス過程回帰みたいにハイパーパラメータの最適化が不要であるモデルでなくてもいいのでは？
→
[?] テスト計算のやり方を変えてみる？
ランダムサンプリングで50点ほどデータ収集してから最適化した場合に、何回で収束するか？をテスト

[?] ベイズ最適化といっても、モデル＋グリッドサーチによる最適化(逆解析)にちょっと変化付けただけ？
獲得関数 = UCBでk=0とすれば、そうなる
[?] k=0を試すとどう変化する？

[?] ベイズ最適化するのに向いてるのは？
    1.探索範囲は狭いが、試行コストが高い
    2.試行コストは安いが、探索範囲が広い
結晶構造の最適化の場合は2.なのでは？
つまり、ランダムサンプリングである程度のデータ数を揃えて置くことは可能
そこから混合ガウスモデル回帰分析をやればいいのでは？
    正規分布の数 = ローカルミニマムの数 とする
    その後、新しいローカルミニマムを発見すれば、そのたびに数を変更する
    
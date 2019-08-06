# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 00:33:45 2019

@author: Akitaka
"""

[1] 機械学習
[1a] サイトで勉強　金子研究室
データ解析に関するいろいろな手法・考え方のまとめ
https://datachemeng.com/summarydataanalysis/


[1a2] twitter
https://twitter.com/hirokaneko226/status/1091116962917244929
    Optimal Additive Kernel (OAK) という、
    サポートベクターマシンにおける新しいカーネルに関する論文。
    変数ごとの値を変換したものを足し合わせることで計算でき、
    Radial Basis Function (RBF) と比べて高速に計算可能。
    データセットによっては RBF の SVM より性能が向上した


https://doi.org/10.1016/j.neucom.2018.10.032
"A new support vector machine with an optimal additive kernel"
Jeonghyun Baek, et al.,
Neurocomputing Volume 329, 15 February 2019, Pages 279-299
    """
    確実に、時間はRBFの方がかかる。OAKは線形カーネルと同程度。
    精度は、一概にどちらが良いとも言えない。他の早いカーネルに比べれば確実に精度は良い
    OAK < RBFなケース
        Fig. 15. Accuracy versus testing time: ‘Two Moons’ dataset.
        Fig. 18. Accuracy versus testing time: ‘Two Spirals’
        Fig. 21. Accuracy versus testing time: ‘Two Circles’.
    OAK > RBFなケース
        Fig. 22. Average accuracy versus testing time: UCI DB. 
        Fig. 23. Average accuracy versus testing time: LIBSVM DB.
    """


[1a3] 遺伝アルゴリズム 石河さん
/home/ishikawa/
    Escp
    GA
        中にEscpが入っている(ホーム直下のと同じ)ことから、
        恐らくは配布用に整理し直している物
    GA-H-16atoms-400GPa.pdf
    GA.tar.gz
    genetic-programing
    programs
    programs.zip

/home/ishikawa/espresso-4.3.1-hybrid/materials/S-O-H/GA-1000GPa/SO4H2/
    make-vcrinput.sh
        #$ -N SO4H2-1000-vcr　とあるので、出力は以下のファイル
        SO4H2-1000-vcr.{e,o,pe,po}86464 
        ~/espresso-4.3.1-hybrid/materials/S-O-H/vcr-SO4H2/
        を作成して、そこへ移動
        ~/espresso-4.3.1-hybrid/materials/S-O-H/GA-1000GPa/SO4H2/results-gen?/
        にある、????.relax_???.inをコピーしていく
        他、色々コピーやディレクトリ作成
        N < N_MAXまでvcrelax計算を繰り返す
        enthalpy.plでエンタルピー計算
            ここで独自のデータベースを使用
            /espresso-4.3.1-hybrid/database/\${element}/enthalpy-\${pp}.data
        update_input.plでvcrelaxのinputを更新？
        各圧力の計算を、このファイルで全部やっている？
        使用ファイル
            res.ecsp
            xargs

    make-QE.sh
        生成物
            results-gen[0-5]
            QE-[1-20].sh
            run.sh

    run.sh: QE-[1-20].shをqsubするスクリプト

    QE-[1-20].sh
        #$ -N J0004　とあるので、出力は以下のファイル
        J00[01-20].{e,o,pe,po}*
        以下を繰り返し
            ${S_NAME}.relax_${STRNUM}.in を作成
            それをインプットとして、relax計算開始
            ${S_NAME}.vcr_${STRNUM}.in を作成
            それをインプットとして、vcrelax計算開始

    rank.sh
        不正終了したジョブの再計算
        エンタルピー計算による、構造のランキング
        使用ファイル
            gen.in
            pop.in
        生成物
            lopt.dat
            res.ecsp.00[0-5] 恐らくは0~5世代の意味
            jobs
        使用外部プログラム
            /home/ishikawa/Ecsp/select.x

    strgen.sh # 次の世代を作成するスクリプト
        生成物 第0世代の時のみ
            gen.in
            par.in
            pop.in
        使用外部プログラム
            /home/ishikawa/Ecsp/rndgen.x 第0世代で使用
            /home/ishikawa/Ecsp/cndgen.x 第1世代以降で使用

    test： 原子位置をまとめたもの？
    xndgen.atp: 原子位置?
    xndgen.cdm: 格子定数?
    xndgen.dat: heredity, mutation, permutaion などの情報
    jobs
        nextの一言のみ
        恐らく、ジョブの進行をコントロールするためのファイル

多分、次のような手順
    make-vcrinput.sh で準備
    strgen.sh # 世代を作成するスクリプト
    make-QE.sh # QE-[1-20].shを作成 & run.shでqsub
    rank.sh　# 結果をランキングに。
    またstrgen.shへいき、繰り返し
    適当なところで終わり。


[1a4] 論文
https://doi.org/10.1103/PhysRevLett.114.105503
"Big Data of Materials Science: Critical Role of the Descriptor"
Luca M. Ghiringhelli, et al.,
Phys. Rev. Lett. 114, 105503 – Published 10 March 2015
    ※160回も引用されている
    記述子とメカニズムの関係が分かっていない場合に、どう記述子を選ぶかが問題
    適切な記述子の選び方を研究する
    対象：二元化合物半導体の結晶構造
        閃亜鉛鉱型またはウルツ鉱型と岩塩型半導体の3つ(=ZB,WZ,RS)
        エネルギー差が非常に小さい。1原子当たりのエネルギー0.001%未満
    model:P(d)
    P_i:物質iのproperty
    d_i:物質iのdescriptor

    先行研究
    1970, Phillips and van Vechten (Ph-vV) [1,2]
    記述子2つ：実験的な誘電率、結晶内での最近接距離、に関連したもの(イコールではない)
    Fig.1： (ZB or WZ) と RS の2クラス分類結果。かなりきれいに分けられている。
    [3–5]:他の記述子を試す。
    ここ最近機械学習がトレンド。マテリアルズインフォマティクス[14–18],バイオ, ケモインフォマティクス[19]
    ほとんどがカーネルリッジ回帰。
    先行研究の記述子は、ad hoc(その場しのぎ)。それがベストだと証明されてない。

    本研究の目的
    ある特定のクラスの物性について、正確な予測ができる記述子を探索するアプローチを提案
    Ph-vVの分類に加えて、ZBとRSの定量的エネルギー差を。

    方法
    特徴量選択にLASSO
    データはDFT+LDA+全電子計算コード FHI-aims
    82化合物。詳しくはSupplemental Material [6]
    孤立中性原子や、2原子分子の特性について計算をした。

    二元系化合物の原子番号ZA,ZBは、最低エネルギー構造を明確に識別する
        多体ハミルトニアン、エネルギーを与えるため
    Fig.2(top):P=ZBとRSのエネルギー差,　d=(ZA,ZB) by KRR
        CVより、予測性能は非常に悪い

    記述子にとって重要な4特性 (if not necessaryとあるが？)
        a.対象材料を一意に特徴づける記述子
        b.全く異なる(似ている)物質は、全く異なる(似ている)記述子によって特徴づけられる
        c.予測したい特性よりも、短い時間で計算できる記述子
        d.記述子の数は可能な限り少なくすること。精度はちゃんと維持した上で。
    Ph-vVのは、cがダメ。原子番号では、b,dがダメ。

    膨大な量の記述子を用意する
    P(d) = d*c 線形モデルにフィッティングして、一番いいdが何かを探す
    次元の数Ωのみがパラメータになっている
    普通に解くことは出来ないので、スパース解が出せるLASSOを使用
    LASSO:特徴量選択の１つ。
    評価：RMSE, maximum absolute error (MaxAE)
    P = ZBとRSのエネルギー差
    
    特徴量作成手順
    単原子(A,B)の特性(2*7=14)：
        イオン化ポテンシャル、電子親和力、highest-occ. E, lowest-unocc. E,
        r_s, r_p, r_d,...(価電子s,p,dの確率密度が最大になる半径)
    二原子分子(AA, BB, AB)の特性(3*3=9)：
        平衡距離、束縛エネルギー、HOMO-LUMO ギャップ
    合計23個
    これを基に、特徴量10000個作成。
    詳しくは、Supplemental Material [6]、Ref.[25]

    LASSOは、記述子の間に相関がある場合に問題
    2つの相関なら計算できても、3つ以上の相関は計算不可能(高コスト)
    1. LASSOで有効変数をpre-select  25~30個
    2. 全探索Subset回帰(Best Subset回帰)  ※読んでもよく分からない

    論文の手続きで見つかった特徴量(3つの合成特徴量の線形回帰)
        [IP(B)-EA(B)]/r_p(A)**2
        |r_s(A)-r_p(B)|/exp[r_s(A)]
        |r_p(A)-r_s(B)|/exp[r_d(A)]

    記述子のロバスト性をチェック
    結果table.1
    RMSEだけでなく、選ばれた記述子も安定
    1つ目の記述子は90%の割合、2つ目の記述子は100%の割合で選ばれた
    
    Zungerによる2D記述子のエラー指標
        Refs. [3,5] and Supplemental Material [6]
        →　小さい　非線形KRRと同程度
    今回の方が、well-definedな手順で作成されており、システマティックに改良できる(次元増加)
    今回の記述子だと、物理的に意味のあるもの(エネルギーギャップなど)を含んでいる
    ただし、AとBの入れ替えに対して対称性はなし
        対称性のある記述子も作ったが、残らなかった。パフォーマンスが悪い
        テストデータを作るときに、対称性が暗黙のうちに壊れた？ EN(A)<EN(B)になった。
    d軌道が3D以上でないと出てこない
    Fig.2(bottom):2D記述子と、ΔE_AB
    上記の4条件を満たしている。a,b,dは作成条件により保証されている

    ロバスト性と物理的意味のチェックのため、P+ノイズでテスト　→　OK
        記述子2つで、93%の割合で識別。 RMSE 10%のみ上昇。詳細はRef.[25]
    
    記述子-特性の関係
    KRRに比べて非常にシンプルなモデルを使った


[1a5] 遺伝アルゴリズム python
20180709
    遺伝アルゴリズムのライブラリー、DEAPを使っているので、インストールした
    pip install -U deap
    ref:
    http://darden.hatenablog.com/entry/2017/04/18/225459
    https://qiita.com/neka-nat@github/items/0cb8955bd85027d58c8e

20180918
    [Pythonコードあり] スペクトル解析における波長領域や時系列データ解析における
    プロセス変数とその時間遅れを選択する方法
    https://datachemeng.com/gawlsgavds/
    遺伝的アルゴリズム (Genetic Algorithm, GA) を使って回帰モデルの推定性能がよくなるように、
    説明変数 (記述子・特徴量・入力変数) を選択する手法を以前解説しました。
    ref:20180709
    
    デモプログラム実行
    https://github.com/hkaneko1985/gawls_gavds
    C:\Users\Akitaka\Downloads\python\0918\gawls_gavds-master
    ただし、時間がかかる
    plsよりはsvrの方が速い？

遺伝的アルゴリズム
http://darden.hatenablog.com/entry/2017/03/29/213948
    外部ライブラリーなしのサンプルプログラム1
    
Pythonで遺伝的アルゴリズム
https://qiita.com/KeisukeToyota/items/0f527a72270430017d8d
    外部ライブラリーなしのサンプルプログラム2

両者の比較 - パラメータ
    n_gene   = 100   # The number of genes.
    =
    gene_length = 10 # 遺伝子長

    n_ind    = 300   # The number of individuals in a population.
    =
    individual_length = 10 # 個体数

    NGEN = 40 # The number of generation loop.
    =
    generation = 20 # 世代数

    MUTPB    = 0.2   # The probability of individdual mutation.
    =
    mutate_rate = 0.1 # 突然変異の確率

    MUTINDPB = 0.05  # The probability of gene mutation.
    CXPB     = 0.5   # The probability of crossover.
    ->(イコールではないが、関係する)
    elite_rate = 0.2 # エリート選択の割合

両者の比較 - 関数
    def create_ind(n_gene):
        """Create a individual."""
    =
    def get_population():
        第1世代の個体群を生成

    def evalOneMax(ind):
        """Objective function."""
    =
    def fitness(pop):
        適応度

    def cxTwoPointCopy(ind1, ind2):
        """Crossover function."""
    =
    def two_point_crossover(parent1, parent2):
        交叉

    def mutFlipBit(ind, indpb):
        """Mutation function."""
        こちらのみ、indpbで変異率を定義
    =
    def mutate(parent):
        突然変異

    対応不明
    def selTournament(pop, n_ind, tournsize):
        """Selection function."""
    =
    def evaluate(pop):
        評価
    +
    アルゴリズム(詳しくは後で)

    def set_fitness(eval_func, pop):
        """Set fitnesses of each individual in a population."""

両者の比較 - アルゴリズム
    # --- Step1 : Create initial generation.
    pop = create_pop(n_ind, n_gene)
    set_fitness(evalOneMax, pop)
    best_ind = max(pop, key=attrgetter("fitness"))
    =
    # 初期個体生成
    pop = evaluate([(fitness(p), p) for p in get_population()])
    print('Generation: 0')
    print('Min : {}'.format(pop[-1][0]))
    print('Max : {}'.format(pop[0][0]))
    print('--------------------------')

    # --- Generation loop.
    print("Generation loop start.")
    print("Generation: 0. Best fitness: " + str(best_ind.fitness))
    for g in range(NGEN):
    =
    for g in range(generation):
        print('Generation: ' + str(g+1))

        # --- Step2 : Selection.
        offspring = selTournament(pop, n_ind, tournsize=3)
    =
        # エリートを選択
        eva = evaluate(pop)
        elites = eva[:int(len(pop)*elite_rate)]

        # --- Step3 : Crossover.
        crossover = []
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            ...
        offspring = crossover[:]
        
        # --- Step4 : Mutation.
        mutant = []
        for mut in offspring:
            ...
        offspring = mutant[:]
        ※offspring:次世代
    =
        # 突然変異、交叉
        pop = elites
        while len(pop) < individual_length:
            if random.random() < mutate_rate:
                m = random.randint(0, len(elites)-1)
                child = mutate(elites[m][1])
            else:
                m1 = random.randint(0, len(elites)-1)
                m2 = random.randint(0, len(elites)-1)
                child = two_point_crossover(elites[m1][1], elites[m2][1])
            pop.append((fitness(child), child))

        # --- Update next population.
        pop = offspring[:]
        set_fitness(evalOneMax, pop)
        # --- Print best fitness in the population.
        best_ind = max(pop, key=attrgetter("fitness"))
        print("Generation: " + str(g+1) + ". Best fitness: "
            + str(best_ind.fitness))
    =
        # 評価
        eva = evaluate(pop)
        pop = eva
        print('Min : {}'.format(pop[-1][0]))
        print('Max : {}'.format(pop[0][0]))
        print('--------------------------')

    print("Generation loop ended. The best individual: ")
    print(best_ind)
    =
    print('Result : {}'.format(pop[0]))

どちらかと言えば、プログラム２の方がわかり易かった


用語
    遺伝子(gene) : 一つの設計変数のこと。
    個体(individual) : 設計変数の1セット。
    個体集合(population) : 個体を集めたセット。
        現世代(population)と次世代(offspring)の2つを用意する必要があります。
    世代(generation) : 現世代と次世代を包括した個体集合の表現。
    適応度(fitness) : 各個体に対する目的関数の値。
    選択(selection) : 現世代から次世代への淘汰のこと。適応度の高いものを優先的に選択します。
    交叉(crossover) : 2個体間の遺伝子の入れ替えのこと。
        生物が交配によって子孫を残すことをモデル化したもの。
    突然変異(mutation) : 個体の遺伝子をランダムに変化させること。

手順
    1.第1世代の個体群を生成
    2.適応度の高い個体（エリート）を選択
        ※プログラム１の方は、トーナメント方と呼ばれるやり方なので、２とは違う
        トーナメント方：「ランダムに一部を選んで、その中から一番いいのを選ぶ。」を繰り返す。
    3.エリートの交叉、突然変異により次世代の個体群を生成
    4.2と3を繰り返し、最後に最も適応度の高い個体を解として出力

遺伝的アルゴリズムによる最適化の現状
京都大学 学術情報メディアセンター  喜多 一
https://www.smapip.is.tohoku.ac.jp/
    ~smapip/2003/tutorial/textbook/hajime-kita.pdf
    理論的な話。2003年当時の状況である点には注意。


システムの最適化
－ ４．遺伝的アルゴリズム（ GA: Genetic Algorithm ） －
https://www.sist.ac.jp/~suganuma/kougi/other_lecture/SE/opt/GA/GA.htm
    大学の教科書的なサイト？
    図を見ながら理解できる
    サンプルプログラムがweb上で動く＆C言語


遺伝的プログラミングによる特徴量生成
https://qiita.com/overlap/items/e7f1077ef8239f454602
    遺伝的プログラミング＝遺伝的アルゴリズムの対象を、数式にしたもの （雑な理解）
    石河さんのやった、Tcの式を遺伝アルゴリズムで計算したのはGPと言える
    選択／交叉／突然変異にはいろいろある
    選択
        トーナメント方式
            集団からランダムに個体を複数抽出し、その中で最も適応度の高い個体を選択する方法
        ルーレット選択
            適応度に応じた確率で個体を選ぶ方法
    交叉
        2点交叉
            交差点をランダムに二つ選び、はさまれている部分を入れ替える方法
        一様交叉
            各要素を独立に1/2の確率で入れ替える方法
    突然変異
        ランダム変異
            個体の遺伝子の一部をランダムに入れ替える方法
        スクランブル
            ランダムに選ばれた2点間の要素の順序をランダムに並べ替える方法

    特徴量設計をGPで行うサンプルプログラムあり
    実行してみた




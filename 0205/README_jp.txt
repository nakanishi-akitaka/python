# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 13:17:09 2019

@author: Akitaka
"""
[1] 機械学習
[1a] サイトで勉強　金子研究室
データ解析に関するいろいろな手法・考え方のまとめ
https://datachemeng.com/summarydataanalysis/
学生・研究者へ
https://datachemeng.com/forstudentsresearchers/


[1a2]
https://twitter.com/hirokaneko226/status/1091475531709657088
    トリパノソーマシステインプロテアーゼ阻害剤のための構造活性相関に関する論文。
    ベンゾイミダゾール誘導体に対して分子動力学シミュレーション。
    熱力学的積分法により結合自由エネルギーを計算したところ実験値と全体的によく一致した

https://doi.org/10.1021/acs.jcim.8b00557
"Understanding Structure–Activity Relationships
 for Trypanosomal Cysteine Protease Inhibitors
 by Simulations and Free Energy Calculations"
Lucianna H. Santos, et al., J. Chem. Inf. Model. 2019,  59, 1, 137-148


https://twitter.com/hirokaneko226/status/1092168739645079552
    Locally Linear Embedding Orthogonal Projection to Latent Structure 
    による異常検出に関する論文。
    プロセス変数Xと製品品質等のYを考慮して、Locally Linear Embeddingで
    X,Yそれぞれ低次元化して、それらの間でPLS。
    Tennessee Eastman プロセスのデータセットで検証

https://doi.org/10.1021/acs.iecr.8b03849
"Quality-Relevant Fault Monitoring Based
 on Locally Linear Embedding Orthogonal Projection to Latent Structure"
J. L. Zhou, el al., Ind. Eng. Chem. Res. 2019,  58, 3, 1262-1272


https://twitter.com/hirokaneko226/status/1092529585902612480
    実験計画法とシミュレーションによるリチウムイオン電池の最適化に関する論文。
    モデルはSingle Particle Model with electrolyte dynamics (SPMe)。
    設計変数は、Bruggeman 係数、カチオン輸率、電解液・陰極・ 陽極における拡散係数、
    陰極・陽極における反応速度定数の７つ

https://doi.org/10.1021/acs.iecr.8b04580
"Optimal Design of Experiments for a Lithium-Ion Cell:
 Parameters Identification of an Isothermal Single Particle Model 
 with Electrolyte Dynamics"
Andrea Pozzi, et al.,
Ind. Eng. Chem. Res. 2019,  58, 3, 1286-1299
    """
    Lithium-ION SIMulation BAttery toolbox(LIONSIMBA)
    →[model order reduction]→
    SPMe → design of experiments → data collection → parameter estimation
     → further experiments → SPMeに戻って、収束するまでループ。
    """


[1a3] 
https://twitter.com/bemoroid/status/1092410408323796992
    BIの実現しない社会、そしてその駄目な社会で飯のために仕方なく働く状態で
    いくらこんなことを言っても、やりがい搾取でしかないんじゃないかなぁ
    
    学生が論文を書くことの、学生にとっての 10 のメリット
    https://datachemeng.com/merits_of_writing_papers_for_students/

[1b] 遺伝的アルゴリズム
[1b1]
既に、遺伝的アルゴリズムを使った結晶構造最適化のフリーソフトはあるのか？
調べたけれども、見つけることはできなかった。
→
USPEXがあった！
http://www.uspex-team.org/en/
http://www.uspex-team.org/en/uspex/overview
実用例
    H3Sもある
他の手法との比較
    USPEX (evolutionary algorithm)
    CALYPSO (PSO method)
    minima hopping method
        for Lennard-Jones clusters of different sizes
        for TiO2 with 48 atoms/cell
    # 私も作るなら、これと比較するべきか

紹介文
https://ma.issp.u-tokyo.ac.jp/app/651


[1b2]
[?] 必要な関数は？
Pythonで遺伝的アルゴリズム
https://qiita.com/KeisukeToyota/items/0f527a72270430017d8d
第1世代の個体群を生成
    def get_population():　
    0,1の二値なので、要変更

適応度
    def fitness(pop):
    ここは外部プログラムでトータルエネルギーを計算するので、要変更というか不要
    サンプルプログラムとしては必要だけれど

評価
    def evaluate(pop):
    トータルエネルギーの低い順番に並べればいいだけなので、降順と昇順を入れ替えるぐらいか
    あるいは、適応度の部分で、エネルギーの正負を入れ替えればそのままでよい。

交叉
    def two_point_crossover(parent1, parent2):
    特に変更するべき点はなし

突然変異
    def mutate(parent):
    0,1の二値なので、要変更

変更点は意外と少ない？


[1b3] 取りあえず、[0,1]の二択から、[0:1]の乱数に変更してみる。
構造最適化においては、乱数と[min,max]から、 (max-min)*乱数+min とすればよい。
    population.append([random.randint(0,1) ...
    →
    population.append([random.random() ...

    child[r] = 1 if child[r]==0 else 0
    →
    child[r] = random.random()

1000世代での結果
Result : (9.87691475509389, 
[0.996831043319251, 0.9842281495565366, 0.9976863877164085, 
0.9958818731000126, 0.9972138920216619, 0.9517144300805206, 
0.9918727872762305, 0.9759084208826739, 0.9978516868283092, 
0.9877260843122857])

十分な結果？

※0以上、1未満なので、1は出ない
http://python-remrin.hatenadiary.jp/entry/2017/04/26/233717
https://note.nkmk.me/python-numpy-random/

後は、外部プログラム(QE)用のスクリプトを作成するプログラムの問題
意外とやることは少ないのか、あるいはこの後こそが問題なのか

[!] 突然変異=初期生成＋交叉でもいい？わざわざそうする意味はともかく
[!] 構造最適化において、最初に置換元となるインプットを用意するのは？
    原子位置、格子定数の部分に、taux1,taux2,taux3,...,やcelldm1,celldm2,...などと書く
    スクリプトで、それらの文字を数値に置換する
    >> 実際にそれでプログラム作成した(後述)


[1b4] 遺伝的アルゴリズム　石河
/home/ishikawa/Ecsp/source/
ライブラリー
    MOD_{gen,lib,par,pop}.f90 >> gen,lib,par,pop

結晶構造を整理する
    cpv2ea.f90 >> cpv2ea.x 
    FINDSYMなどで代用可能？

ランダムな結晶構造を作成する
    rndgen.f90 >> rndgen.x
    手順
    1.乱数の配列random(1000)を作成
    2.乱数の配列の先頭から使用して、a,b,cos(γ),cos(α),cos(β),cに当てている 
        cc(1), cc(2), cc(3) = cx, cy, cz
        xi0, et0がよく分からない
        たぶん、xi0, et0 = 格子定数α, β用の乱数 

        cosg = cos((acos(MaxAng)-acos(MinAng))*random(ii) + acos( MinAng))
        xi0 = random(ii)
        et0 = random(ii)
        のように、αβγで生成方法が異なる。おそらくは、αβは0~90までなので、cos=0~1ということ。
        
        妙に回りくどい乱数生成方法のようにも見える。
        abcαβγを全部乱数で直接表してはだめなのか？
        ベルトランの逆説 >> 「無作為に弦を選ぶ方法は複数ある」から考えると、
        「無作為に平行六面体(=unit cell)を選ぶ方法は複数ある」と言えるのかもしれない

    出力ファイル 
        xndgen.{dat,cmd,atp},test
        =
        xndgen.dat: heredity, mutation, permutaion などの情報
        xndgen.cdm: 格子定数 = (cdm(j,i), j=1,6), b/a, c/a
        xndgen.atp: 原子位置 =  xi(j,i), et(j,i), zt(j,i), &
            &   xi(j,i)*a + et(j,i)*b*cosg + zt(j,i)*c*cosb, &
            &               et(j,i)*b*sing + zt(j,i)*c*(cosa-cosb*cosg)/sing,&
            &                                zt(j,i)*c*p/sing
        test： xi0, et0, cc(1), cc(2), c, cosg, cosa, cosb

次世代の結晶構造を作成する
    cndgen.f90 >> cndgen.x
    iop:1 HEREDITY    :2 MUTATION     :3 PERMUTATION
    各操作は外部ライブラリー
    heredity   ( cdm0, xi0, et0, zt0, indiv1, indiv2, i ) << vo_her.f90
        遺伝させるだけなのでシンプル、かと思ったら複雑
        恐らくは、2つの構造を足し合わせている。
        これこそ交配なのでは？
    mutation   ( cdm0, xi0, et0, zt0, indiv1 ) << vo_mut.f90
        ax,ay,az,...,czの9成分全てに乱数の影響を与えている
    permutation( cdm0, xi0, et0, zt0, indiv1 ) << vo_per.f90
        ランダムに選んだ2つの個体の遺伝子xi0,et0,zt0を置換している
        格子定数はそのままコピー
    その後、chk_hc.f90 >> chk を呼び出し
        MinDis,MinLat,MinAng,MaxAngなどの条件を満たしているかチェック

    出力ファイル 
        xndgen.{dat,cmd,atp},test
        =
        xndgen.dat: heredity, mutation, permutaion などの情報
        xndgen.cdm: 格子定数 =  cdm0, b/a, c/a
        xndgen.atp: 原子位置 = xi0(j), et0(j), zt0(j), &
            & xi0(j)*a + et0(j)*b*cosg + zt0(j)*c*cosb, &
            &            et0(j)*b*sing + zt0(j)*c*(cosa-cosb*cosg)/sing, &
            &                            zt0(j)*c*p/sing
        test: cdm0(1)  cdm0(2)  cdm0(3)  cdm0(4)  cdm0(5)  cdm0(6) Volume?


なんか、思っていたのとだいぶ違う。
これは、
ref:20190201
    遺伝的プログラミングによる特徴量生成
    https://qiita.com/overlap/items/e7f1077ef8239f454602
にある通り、選択／交叉／突然変異の方法にもいろいろあることが原因だろう
私がサンプルとして選んだ
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

[1b5] SSCHA
ふと考えると、SSCHAで既に、「原子位置を生成 >> ジョブにする >> 結果を集計」
という手順で計算するスクリプトやプログラムは経験済み
それを応用すればいいのでは？

[1b6] QEの仕様
原子位置は、+-/*を使える。celldm(1~6)は無理

[1b7] アイディア
vcrelaxでなく、relax
    これは特に意味ないか

niterを敢えて少なめに設定するのは？
    エネルギーがあまり低くない構造を時間かけて最小化するより、
    さっさと見切りをつけた方が効率的なのではないか
    どの操作によって生成した構造であれ、それがグローバルミニマムの近くにあれば、
    十分にエネルギーは低いハズ


[1b8] プログラミング
とりあえず、ランダム生成したパラメータでscf.jobを投入するスクリプトまでは完成
vclreaxはしない
~/GA/step0_parameters.sh
~/GA/step1_0th_generation.sh
ランダムサーチほぼ完成といえる


[1c] ランダム探索
ref:20170904の添付のメモ帳
下司先生、シリコン２，４原子などでテスト


[1d] ベイズ最適化
遺伝的アルゴリズムであれ、これであれ、結局は
原子位置を生成 >> ジョブにする >> 結果を集計」という手順にすぎない
そう考えると、ほぼ同じようなファイル構成にして、プログラムを組むことができるのではないか





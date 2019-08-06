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



[1b] 遺伝的アルゴリズム
[1b1] プログラミング
0214 C4つで第0世代から計算し直し＋スクリプトアップデート, 現在第0-1世代
0215 第2-9世代で終了。また第0からやり直し。
    またsortがおかしい？毎回ファイルに出力させてチェックすることにした

[?][!] 同じ構造の固体が出来てしまった場合、選別した者が同じものばかりになってしまう！
    どのように解決する？手動変更というゴリ押しもあるが
    =>
    difference()関数で判定する方式

[!] 一部でNaNが出てしまう！
    cat indv*/fit_gene > next.in
    =>
    grep -v "NaN" indv*/fit_gene > next.in

[?] そもそも、最初を完全にランダムにする必要があるのか？
    格子定数や原子位置の組み合わせとして、有名なもの、対称性の良いものも
    ランダムなものと混ぜたらいいのでは？




[1c] 遺伝的プログラミング
[todo]
    遺伝的プログラミングによる特徴量生成
    https://qiita.com/overlap/items/e7f1077ef8239f454602
    ->Tc計算式そのものもGAでやる(石河さんと同じこと)

[1c1]
0214 Tcに応用 kNN, SVR 評価 R^2など
0215 Tc計算=>make_regressionに戻す
そこで、OLS, kNN, SVR(ハイパーパラメータはデフォルト)を比較すると？
評価 MAE, 「交叉確率50%、突然変異確率10%、10世代まで進化」×1

OLS
### Results
Baseline MAE train : -1.993516463016931e-13
Baseline MAE test : -2.673417043297377e-13
Best MAE train : -9.734435479913372e-14
Best MAE test : -1.5916157281026243e-13

### Generated feature expression
add(cos(sin(mul(ARG9, ARG5))), sin(mul(ARG9, cos(neg(ARG6)))))


kNN
### Results
Baseline MAE train : -86.87009170893735
Baseline MAE test : -98.83686947523593
Best MAE train : -55.93802925477928
Best MAE test : -78.01952391925586

### Generated feature expression
add(ARG6, add(add(add(ARG3, ARG0), ARG7), ARG7))


SVR
### Results
Baseline MAE train : -141.5247537041722
Baseline MAE test : -141.3942953635643
Best MAE train : -139.45223151845084
Best MAE test : -141.06337059783655

### Generated feature expression
add(add(add(ARG6, ARG7), ARG4), add(sin(add(ARG7, add(ARG3, ARG6))), ARG0))


SVR(kernel='linear')
### Results
Baseline MAE train : -110.5835534487372
Baseline MAE test : -134.30407096388254
Best MAE train : -0.14761104064917335
Best MAE test : -94.9618152156369

### Generated feature expression
add(ARG3, add(add(add(ARG6, ARG0), ARG4), ARG7))

文字通り、OLS以外は、MAEがけた違いに悪い
学習データがたまたま線形手法に向いていただけ？
    SVRだと特徴量作成しても大差ないのに、SVR(kernel='linear')だと劇的に改善するため
    実際、make_regressionは、線形のものしか作れないっぽい

=>
[1c2] sin(x)+ノイズ
2018/0619/20180613test0.py より、sin(x)+ノイズをサンプルにしてみる

OLS
### Results
Baseline MAE train : -0.440805858118677
Baseline MAE test : -0.43888154124948126
Best MAE train : -0.2233213523203633
Best MAE test : -0.3322370359496641

### Generated feature expression
sub(sin(sin(sin(ARG0))), sin(add(ARG0, ARG0)))
※これで１つの変数

kNN
### Results
Baseline MAE train : -0.317071852963461
Baseline MAE test : -0.4539328918849407
Best MAE train : -0.2897236150843182
Best MAE test : -0.4717038746130493

### Generated feature expression
protectedDiv(neg(tan(ARG0)), add(add(ARG0, sub(add(ARG0, ARG0), neg(ARG0))), 
    cos(sin(add(ARG0, ARG0)))))
※これで１つの変数


SVR
### Results
Baseline MAE train : -0.21035451231184493
Baseline MAE test : -0.26292652136091527
Best MAE train : -0.19233917037568263
Best MAE test : -0.28222863951490307

### Generated feature expression
cos(sub(sin(sin(mul(ARG0, ARG0))), ARG0))

どれも微妙？


ノイズを+-3 -> +-0.5に減らす
OLS
### Results
Baseline MAE train : -0.38005970281182105
Baseline MAE test : -0.4026924764660472
Best MAE train : -0.07145036186466826
Best MAE test : -0.07406321319108752

### Generated feature expression
sub(sub(cos(cos(ARG0)), sin(ARG0)), sin(add(ARG0, ARG0)))
 = cos(cos(x))-sin(x) - sin(x+x)

kNN
### Results
Baseline MAE train : -0.11730483355904145
Baseline MAE test : -0.18944421566441372
Best MAE train : -0.10942824722161333
Best MAE test : -0.4003057675740476

### Generated feature expression
add(cos(ARG0), sub(neg(sin(add(ARG0, ARG0))), mul(ARG0, ARG0)))
 = cos(x)+(-sin(x+x)-x*x)

SVR
### Results
Baseline MAE train : -0.085517575739754
Baseline MAE test : -0.11384451831199596
Best MAE train : -0.07852344100101391
Best MAE test : -0.12723685516518207

### Generated feature expression
sin(sub(neg(ARG0), mul(ARG0, ARG0)))
 = sin(-x-x*x)



いっそ、ノイズなしに
OLS
### Results
Baseline MAE train : -0.36218102555584597
Baseline MAE test : -0.3611586710256758
Best MAE train : -1.8153881176097287e-16
Best MAE test : -2.997602166487923e-16

### Generated feature expression
add(mul(neg(ARG0), sub(ARG0, ARG0)), sub(ARG0, sin(ARG0)))
= -x*(x-x)+(x-sin(x))

kNN
### Results
Baseline MAE train : -0.046362264165211366
Baseline MAE test : -0.17143307825309884
Best MAE train : -0.03894652118982389
Best MAE test : -0.21349449578772325

### Generated feature expression
add(sub(sin(ARG0), cos(neg(add(ARG0, ARG0)))), sin(add(ARG0, sin(ARG0))))
= sin(x)-cos(-x-x)+sin(x+sin(x))

SVR
### Results
Baseline MAE train : -0.0777565932822952
Baseline MAE test : -0.0916386470903859
Best MAE train : -0.0452934206332536
Best MAE test : -0.15877510795313798

### Generated feature expression
cos(protectedDiv(tan(sin(ARG0)), cos(ARG0)))
= cos(tan(sin(x))/cos(x))


[1c3] サンプル数変える
サンプル数100 -> 1000
OLS
### Results
Baseline MAE train : -0.3508669014377877
Baseline MAE test : -0.38123607601044307
Best MAE train : -2.307298774782695e-16
Best MAE test : -1.587608083192249e-16

### Generated feature expression
sin(ARG0)
sin(x)がやっと出た！
やはりというか、ある程度のサンプル数が必要だった様子


サンプル数1000+ノイズ0.1
OLS
### Results
Baseline MAE train : -0.38655019509768024
Baseline MAE test : -0.39922165311776864
Best MAE train : -0.14202563675430313
Best MAE test : -0.14802659393622428

### Generated feature expression
sin(tan(sin(sin(ARG0))))


サンプル数1000+ノイズ0.01
OLS
### Results
Baseline MAE train : -0.3948273348453206
Baseline MAE test : -0.41055462411619714
Best MAE train : -0.1670740079338217
Best MAE test : -0.13888920057979304

### Generated feature expression
add(sin(ARG0), protectedDiv(ARG0, ARG0))
= sin(x)+x/x

サンプル数1000+ノイズ0.001
OLS
### Results
Baseline MAE train : -0.4011359312698513
Baseline MAE test : -0.4066011331913849
Best MAE train : -0.16670673764883567
Best MAE test : -0.1580725428714733

### Generated feature expression
sin(ARG0)

よって、ノイズには相当左右される


「サンプル数1000+ノイズなし」で、OLS以外は？
kNN
### Results
Baseline MAE train : -0.003728577976571887
Baseline MAE test : -0.014421184996387717
Best MAE train : -0.003688012841438213
Best MAE test : -0.01837659317892765

### Generated feature expression
mul(mul(tan(ARG0), sin(sin(ARG0))), neg(tan(ARG0)))

SVR
### Results
Baseline MAE train : -0.07700341054181895
Baseline MAE test : -0.0771908383392787
Best MAE train : -0.0351777217514168
Best MAE test : -0.03715360363600813

### Generated feature expression
sin(cos(add(tan(neg(ARG0)), cos(tan(ARG0)))))

Ridge
### Results
Baseline MAE train : -0.3578361429388403
Baseline MAE test : -0.38678582404700035
Best MAE train : -0.0008618216489479861
Best MAE test : -0.003342280556391265

### Generated feature expression
add(sin(ARG0), sin(ARG0))
実質正解

LASSO
### Results
Baseline MAE train : -0.660531334823314
Baseline MAE test : -0.6531643329197354
Best MAE train : -0.16572959850532992
Best MAE test : -0.17362894914383697

### Generated feature expression
add(add(add(add(sin(ARG0), ARG0), add(sin(ARG0), sin(ARG0))), 
add(sin(ARG0), add(sin(ARG0), sin(ARG0)))), sin(ARG0))

EN
### Results
Baseline MAE train : -0.4611768016768538
Baseline MAE test : -0.4811055369073737
Best MAE train : -0.08876262930635176
Best MAE test : -0.08825914638608971

### Generated feature expression
add(add(add(add(sin(ARG0), sin(ARG0)), sin(ARG0)), sin(ARG0)), 
add(add(sin(ARG0), sin(ARG0)), add(sin(ARG0), sin(ARG0))))

単純に線形であればいいというものでもないらしい



[1c4]
OLS, Ridgeで同様のテスト
cos(x) 

OLS
### Results
Baseline MAE train : -0.6269345354793601
Baseline MAE test : -0.6084889862433223
Best MAE train : -3.0712953881517624e-16
Best MAE test : -1.125748799735149e-16

### Generated feature expression
cos(ARG0)

Ridge
### Results
Baseline MAE train : -0.6195049897663281
Baseline MAE test : -0.597433040658766
Best MAE train : -0.002056215720140516
Best MAE test : -0.00819921455040715

### Generated feature expression
cos(ARG0)


tan(x) 

OLS
### Results
Baseline MAE train : -5.3290199833686165
Baseline MAE test : -5.414165592001782
Best MAE train : -1.6313469672346593e-15
Best MAE test : -2.0053837063160886e-15

### Generated feature expression
sub(neg(ARG0), tan(ARG0))


Ridge
### Results
Baseline MAE train : -5.378892487221861
Baseline MAE test : -6.39338021632077
Best MAE train : -1.3157763834389896e-05
Best MAE test : -0.0001005904342810161

### Generated feature expression
tan(ARG0)


以上のように、だいたいridgeが良かった
しかし、sin(x)*cos(x)のようなものは無理だったので限界はある
ノイズなしの三角関数でもそう簡単にはいかないあたり、Tc計算に応用して、いい公式が出せるのか疑問

もっとも、直接でなくても、GPで作った特徴量で再びモデル作成(GPに使ったのとは別の手法)、
という手段はもちろんあり得る。

また、別にGPでやるのがいつでもベストな訳ではない
あくまで一つの選択肢を覚えたに過ぎないことに注意


[1c5]
遺伝的プログラミングによる特徴量生成
https://qiita.com/overlap/items/e7f1077ef8239f454602
統計モデル精度向上には、目的変数の特徴を上手く表現する変数を見つけることが重要

特徴量の探索
    ベースとなる説明変数を用意した上でそれらの交互作用項や対数変換項などを特徴量に加え、
    変数選択や次元圧縮をするという方法検証する方法
        変数が膨大になりやすく総当り的な探索が難しくなる
        次元圧縮の手法選択やパラメータ調整も重要になるので工数が膨大になる

    遺伝的プログラミングを用いてモデル精度を向上させるような変数を逐次的に生成し、
    段階的に精度を向上させる方法
        既存の説明変数セットから単純にモデル精度を上げるような変数を生成したい場合に、
        それなりに使えるのではないか

        ただし進化の過程で非常に膨大なモデルを学習する必要があるので、
        データサイズや特徴量数が大きい場合や複雑なモデルを使いたい場合は、計算コストが苦しい。
        今回はロジスティック回帰モデルを使っていましたが、
        それでも説明変数が増えるに従って指数関数的に速度が遅くなっていくので、
        20変数の生成でも数時間かかってしまいました。
        遺伝的プログラミングのパラメータも結構な数があるので、
        場合によっては調整に時間がかかることも懸念されます。

        実際に利用する際はサンプリングでデータサイズを減らしたり、
        一部の特徴量のみを利用するなどの工夫を行った方が良いかもしれません。


[1c6] GAの参考リンク
DEAP
http://darden.hatenablog.com/entry/2017/04/18/225459
    ライブラリーDEAPの使い方

Pythonの進化計算ライブラリDeap
https://qiita.com/neka-nat@github/items/0cb8955bd85027d58c8e
    ライブラリーDEAPの使い方

RRLの学習にGAを使ってみる
http://darden.hatenablog.com/entry/2017/05/02/183447
    GAを機械学習の重み最適化に使う例

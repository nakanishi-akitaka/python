# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 00:20:17 2019

@author: Akitaka
"""

[1b] ベイズ最適化
[?] テスト計算にするのに、なんの関数がいいのか？
http://www.techscore.com/blog/2016/12/20/機械学習のハイパーパラメータ探索-ベイズ最適/
ここでは、「x * sin(x) の最大化」
検索した結果、ベンチマークがかなり多数あることを発見した

最適化アルゴリズムを評価するベンチマーク関数まとめ
https://qiita.com/tomitomi3/items/d4318bf7afbc1c835dda
合計34個

    余談だが、役に立ちそうなもの
    Qiitaの数式チートシート
    https://qiita.com/PlanetMeron/items/63ac58898541cbe81ada

Test functions for optimization
https://en.wikipedia.org/wiki/Test_functions_for_optimization
合計42個
ただし、種類が色々
single-objective optimization
    単純に１つの関数を最適化する問題

constrained optimization
    おそらく、拘束条件つきの最適化

multi-objective optimization
    おそらく、複数の関数を同時に最適化する問題(※)

※
多目的最適設計 (multiobjective optimization) 
http://jp.midasuser.com/cadrobo/dictionary_view.asp?idx=1189
    製品やシステムの性能を向上させる場合には、目標とする性能が複数ある場合がほとんどです。
    例えば、電子機器のエネルギー消費率を高めると同時に、価格と製品サイズを減らそうとする場合、
    目標とする性能は、三つになります。
    このように、製品やシステムの機能を同時に向上させようとする
    最適設計(optimum design)の問題を多目的最適設計と呼びます。

[1b2-1b5]
※特筆しない限り、以下の条件で固定
    獲得関数 = UCB(k=sqrt(100)=10)
    探索範囲 [x,y] = [-10,-10],...,[10,10]
    試行回数 n_iteration = 50

[1b2] ベイズ最適化のサンプル Ackley function
y = 20 - 20*exp(-0.2*sqrt(x**2)) + e - exp(cos(2*pi*x))
iter = 35回で、x=0.131072に収束。
厳密解は x = 0 なので少しずれた。x_gridにはあるので、なぜそこに行かなかったのか？
σを考慮に入れる限り、少しずれたとこに行くせいかもしれない

[1b2] ベイズ最適化のサンプル Sphere function
y=x**2
iter = 50回で、x=0に収束。
厳密解は x = 0 なのでバッチリ。単純な二次関数なので当然だが。
一方で、モデルが凸凹だらけの歪なものになってしまっているのが気になる。

[1b3] ベイズ最適化のサンプル Rosenbrock function
y = Sum_i=1^n-1 (100*(x_i+1 - x_i**2)**2 + (x_i - 1)**2)
iter = 23回で、x = (0.4, 0.4) に収束。
厳密解は (1,1) なのでややずれる。
gridは、0, 0.2, 0.4, 0.6, 0.8, 1.0, ... なので、そのせいではない。



※入力を2次元に拡張する部分で大苦戦
ref:2次元格子点の生成
pythonで格子点を生成する
https://qiita.com/hitoshi_xyz/items/ec82e108c3ec827512a3

【Python】ふたつの配列からすべての組み合わせを評価
http://kaisk.hatenadiary.com/entry/2014/11/05/041011

[Perl][Python]格子点の生成
http://d.hatena.ne.jp/odz/20070131/1170284561

【numpy】meshgrid メモ
http://umashika5555.hatenablog.com/entry/2017/10/26/043348

配列の要素から格子列を生成するnumpy.meshgrid関数の使い方
https://deepage.net/features/numpy-meshgrid.html

ref:2次元格子点に要素を追加
多次元配列の結合を行うオブジェクトnp.c_とnp.r_の使い方
https://deepage.net/features/numpy-cr.html


[1b4] ベイズ最適化のサンプル Ackley function 2D
y = 20 - 20*exp(-0.2*sqrt((x1**2+x2**2)*0.5)
     + e - exp((cos(2*pi*x1)+cos(2*pi*x2))*0.5)
iter = 50回で、x = [[-7.6  5. ]]収束とは言い難い
    200まで増やしても収束はせず
厳密解は x = (0,0) なのでだいぶずれた。

[1b5] ベイズ最適化のサンプル Sphere function 2D
y = x1**2 + x2**2
iter = 73回で、x = [[0. 0.]] に収束。
厳密解は x = (0, 0) なのでバッチリ。単純な二次関数なので当然だが。




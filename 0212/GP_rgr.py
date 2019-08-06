# -*- coding: utf-8 -*-
"""

遺伝的プログラミングによる特徴量生成
https://qiita.com/overlap/items/e7f1077ef8239f454602

Created on Tue Feb 12 16:02:54 2019
@author: Akitaka
"""

import operator, math, random, time
import numpy as np

from deap import algorithms, base, creator, tools, gp

from sklearn.datasets        import make_regression
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics         import mean_absolute_error

import matplotlib.pyplot as plt

# サンプルデータの生成
X, y = make_regression(n_samples=100, n_features=10,
                       n_informative=5, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

# ベースラインスコアの算出

rgr = LinearRegression()
base_train_mae = np.mean(cross_val_score(
        rgr, X_train, y_train, scoring="neg_mean_absolute_error", cv=5))
rgr.fit(X_train, y_train)
y_incv = cross_val_predict(rgr, X_test, y_test, cv=5)
base_test_mae = -mean_absolute_error(y_test, y_incv)

# 除算関数の定義
# 左項 / 右項で右項が0の場合1を代入する
def protectedDiv(left, right):
    eps = 1.0e-7
    tmp = np.zeros(len(left))
    tmp[np.abs(right) >= eps] = left[np.abs(right) >= eps] / right[np.abs(right) >= eps]
    tmp[np.abs(right) < eps] = 1.0
    return tmp


# 乱数シード
random.seed(123)

# 適合度を最大化するような木構造を個体として定義
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# 初期値の計算
# 学習データの5-fold CVのAUCスコアを評価指標の初期値とする
n_features = X_train.shape[1]
rgr = LinearRegression()
prev_mae = np.mean(cross_val_score(
        rgr, X_train, y_train, scoring="neg_mean_absolute_error", cv=5))

# メインループ
# resultsに特徴量数、学習データのAUCスコア（5-fold CV）、テストデータのAUCスコアを保持する
# exprsに生成された特徴量の表記を保持する
results = []
exprs = []
for i in range(100):
    # 構文木として利用可能な演算の定義
    pset = gp.PrimitiveSet("MAIN", n_features)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(np.cos, 1)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.tan, 1)

    # 関数のデフォルト値の設定
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # 評価関数の設定
    # 新しく生成した変数を元の変数に加えて5-fold CVを求める
    def eval_genfeat(individual):
        func = toolbox.compile(expr=individual)
        features_train = [X_train[:,i] for i in range(n_features)]
        new_feat_train = func(*features_train)
        X_train_tmp = np.c_[X_train, new_feat_train]
        return np.mean(cross_val_score(rgr, X_train_tmp, y_train, scoring="neg_mean_absolute_error", cv=5)),

    # 評価、選択、交叉、突然変異の設定
    # 選択はサイズ10のトーナメント方式、交叉は1点交叉、突然変異は深さ2のランダム構文木生成と定義
    toolbox.register("evaluate", eval_genfeat)
    toolbox.register("select", tools.selTournament, tournsize=10)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # 構文木の制約の設定
    # 交叉や突然変異で深さ5以上の木ができないようにする
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5)) 

    # 世代ごとの個体とベスト解を保持するクラスの生成
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    # 統計量の表示設定
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # 進化の実行
    # algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 10, 
    # 交叉確率50%、突然変異確率10%、10世代まで進化
    start_time = time.time()
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 10, stats=mstats, halloffame=hof, verbose=True)
    end_time = time.time()

    # ベスト解とAUCの保持
    best_expr = hof[0]
    best_mae = mstats.compile(pop)["fitness"]["max"]

    # 5-fold CVのAUCスコアが前ステップのAUCを超えていた場合
    # 生成変数を学習、テストデータに追加し、ベストAUCを更新する
    if prev_mae < best_mae:
        # 生成変数の追加
        func = toolbox.compile(expr=best_expr)
        features_train = [X_train[:,i] for i in range(n_features)]
        features_test = [X_test[:,i] for i in range(n_features)]
        new_feat_train = func(*features_train)
        new_feat_test = func(*features_test)
        X_train = np.c_[X_train, new_feat_train]
        X_test = np.c_[X_test, new_feat_test]

        ### テストAUCの計算（プロット用）
        rgr.fit(X_train, y_train)
        y_incv = cross_val_predict(rgr, X_train, y_train, cv=5)
        train_mae = -mean_absolute_error(y_train, y_incv)
        y_incv = cross_val_predict(rgr, X_test, y_test, cv=5)
        test_mae = -mean_absolute_error(y_test, y_incv)

        # ベストAUCの更新と特徴量数の加算
        prev_mae = best_mae
        n_features += 1

        # 表示と出力用データの保持
        print(n_features, best_mae, train_mae, test_mae, end_time - start_time)
        results.append([n_features, best_mae, train_mae, test_mae])
        exprs.append(best_expr)

        # 変数追加後の特徴量数が30を超えた場合break
        if n_features >= 30:
            break

# 結果の出力
print()
print("### Results")
print("Baseline MAE train :", base_train_mae)
print("Baseline MAE test :", base_test_mae)
print("Best MAE train :", results[-1][1])
print("Best MAE test :", results[-1][3])

# 結果のプロット
res = np.array(results)
plt.plot(res[:,0], res[:,1],"o-", color="b", label="train(5-fold CV)")
plt.plot(res[:,0], res[:,3],"o-", color="r", label="test")
plt.plot(10, base_train_mae, "d", color="b", label = "train baseline(5-fold CV)")
plt.plot(10, base_test_mae, "d", color="r", label = "test baseline")
plt.xlim(9,31)
plt.grid(which="both")
plt.xlabel('n_features')
plt.ylabel('MAE')
plt.legend(loc="lower right")
plt.savefig("gp_featgen.png")

# 生成した構文木の出力
print()
print("### Generated feature expression")
for expr in exprs:
    print(expr)

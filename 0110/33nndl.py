# -*- coding: utf-8 -*-
"""
3.3 ニューラルネットワーク(ディープラーニング)
https://nozma.github.io/ml_with_python_note/
    3-3-ニューラルネットワークディープラーニング.html

Created on Thu Jan 10 13:29:43 2019

@author: Akitaka
"""

import mglearn
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=100, noise=.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(
  X, y, stratify=y, random_state=42
)
mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

#%%
"""
デフォルトではMLPは100のノードからなる単一の隠れ層を持つが、
これは小さなデータセットに対しては大きすぎるので10に減らしてみる。
"""
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
"""
上記の例で境界がギザギザなのは、デフォルトで活性化関数がrelu関数のため。
層を増やしたり、活性化関数にtanhを用いることで境界を滑らかにできる。
"""

#%%
"""
まず隠れ層を1層ふやしてみる。
"""
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

#%%
"""
さらに活性化関数にtanhを指定する。
"""
mlp = MLPClassifier(solver='lbfgs', activation='tanh',
                    random_state=0, hidden_layer_sizes=[10, 10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
"""
ニューラルネットワークにはまだまだパラメータがある。
重みに対してL2正則化を行うことができる。デフォルトでは正則化は非常に弱い。
"""
#%%
"""
以下は10ノードと100ノードの2層の隠れ層を持つニューラルネットワークに対し、
L2正則化の程度を調整するパラメータalphaを変えた効果を示している。
"""
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for axx, n_hidden_nodes in zip(axes, [10, 100]):
  for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
    mlp = MLPClassifier(solver='lbfgs', random_state=0,
                        hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],
                        alpha=alpha)
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train ,ax=ax)
    ax.set_title("隠れ層=[{}, {}]\nalpha={:.4f}".format(
                 n_hidden_nodes, n_hidden_nodes, alpha))

#%%
"""
ニューラルネットワークは重みの初期値を乱数で決めるが、
この影響は小さいネットワークでは大きく現れることがある。
"""
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for i, ax in enumerate(axes.ravel()):
  mlp = MLPClassifier(solver='lbfgs', random_state=i,
                      hidden_layer_sizes=[100, 100])
  mlp.fit(X_train, y_train)
  mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
  mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)

#%%
"""
次に、実データとしてcancerを使ってニューラルネットワークを適用してみる。
cancerはデータセットのレンジが非常に幅広いデータである。
まずはデータセットそのままでニューラルネットワークを適用する。
"""
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
  cancer.data, cancer.target, random_state=0
)
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print("訓練セットの精度: {:.2f}".format(mlp.score(X_train, y_train)))
## 訓練セットの精度: 0.91
print("テストセットの精度: {:.2f}".format(mlp.score(X_test, y_test)))
## テストセットの精度: 0.88

#%%
"""
精度は良いもののさほどではない。MLPはデータのスケールが同じくらいであることが望ましい。
また、平均が0で分散が1であれば理想的である。
そのような変換をここでは手作業で行う
(StandardScalerを使えばもっと簡単にできるが、これは後に説明される)。
"""

mean_on_train = X_train.mean(axis=0) # 各データセットの平均値
std_on_train = X_train.std(axis=0) # 各データセットの標準偏差
# 平均を引いてスケーリングする
X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train
# MLPを適用
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)
## C:\Users\Akitaka\Anaconda3\lib\site-packages\sklearn\neural_network
## \multilayer_perceptron.py:564: ConvergenceWarning:
## Stochastic Optimizer: Maximum iterations (200)
## reached and the optimization hasn't converged yet.
##  % self.max_iter, ConvergenceWarning)
print("訓練セットの精度: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
## 訓練セットの精度: 0.991
print("テストセットの精度: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
## テストセットの精度: 0.965

#%%
""" 
これで精度はグッと良くなったが、収束に関する警告が出ている。
繰り返し数が不足しているので、max_iterパラメータを通じて繰り返し数を増やす。
"""
mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, y_train)
print("訓練セットの精度: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
## 訓練セットの精度: 0.993
print("テストセットの精度: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
## テストセットの精度: 0.972

#%%
"""
訓練セットに対する精度は上がったが、汎化性能があまり変化しない。
パラメータalphaを大きくして、正則化を強くし、モデルを単純にするともっと汎化性能が上がるかもしれない。
"""
mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)
print("訓練セットの精度: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
## 訓練セットの精度: 0.988
print("テストセットの精度: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
## テストセットの精度: 0.972

#%%
"""
ニューラルネットワークの解析は線形モデルや決定木に比べると難しい。
隠れ層における重みを可視化するという手があるので以下に示す。
"""
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("重み行列の列")
plt.ylabel("特徴量")
plt.colorbar()





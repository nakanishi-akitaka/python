# -*- coding: utf-8 -*-
"""

ベイズ最適化(GP-UCB アルゴリズム)による探索.
環境
  Python 3
  scikit-learn==0.18.1
  matplotlib==1.5.3
  numpy==1.11.2

http://www.techscore.com/blog/2016/12/20/機械学習のハイパーパラメータ探索-ベイズ最適/
+
https://qiita.com/tomitomi3/items/d4318bf7afbc1c835dda

Created on Thu Jan  3 14:44:04 2019

@author: Akitaka
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
 
 
def blackbox_func(x):
    """
    Ackley function
    https://qiita.com/tomitomi3/items/d4318bf7afbc1c835dda
    もともとは最小化問題なのでマイナスをかける
    """
    y = 20 - 20 * np.exp(-0.2*np.sqrt(x**2)) + np.e - np.exp(np.cos(2*np.pi*x))
    return -y 
 
def acq_ucb(mean, sig, beta=3):
    """
    獲得関数 (Upper Confidence Bound)
    $ x_t = argmax\ \mu_{t-1} + \sqrt{\beta_t} \sigma_{t-1}(x) $
    """
    acq = mean + sig * np.sqrt(beta)
    return np.argmax(acq)
 
 
def plot(x, y, X, y_pred, sigma, title=""):
 
    plt.figure(figsize=(6,3))
    plt.plot(x, blackbox_func(x), 'r:', \
             label=u'$blackbox func(x) = x\,\sin(x)$')
    plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
    plt.plot(x, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.96 * sigma,(y_pred + 1.96 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.show()
 
 
 
   
# アプリケーションエントリポイント
if __name__ == '__main__':
    
    # パラメータの取りうる範囲
    x_min = -32.768
    x_max =  32.768
    x_grid = np.atleast_2d(np.linspace(x_min, x_max, 1001)[:1000]).T
    print(x_grid[500],blackbox_func(x_grid[500]))
    
    # 初期値として x=x_min+1, x_max-1 の 2 点の探索をしておく.
    X = np.atleast_2d([x_min+1, x_max+1]).T
    y = blackbox_func(X).ravel()
 
    
    # Gaussian Processs Upper Confidence Bound (GP-UCB)アルゴリズム
    # --> 収束するまで繰り返す(収束条件などチューニングポイント)
    n_iteration = 50
    for i in range(n_iteration):
    
        # 既に分かっている値でガウス過程フィッティング
        # --> カーネル関数やパラメータはデフォルトにしています(チューニングポイント)
        gp = GaussianProcessRegressor()
        gp.fit(X, y)
        
        # 事後分布が求まる
        posterior_mean, posterior_sig = gp.predict(x_grid, return_std=True)
        
        # 目的関数を最大化する x を次のパラメータとして選択する
        # --> βを大きくすると探索重視(初期は大きくし探索重視しイテレーションに同期して減衰させ
        # 活用を重視させるなど、チューニングポイント)
        idx = acq_ucb(posterior_mean, posterior_sig, beta=100.0)
        x_next = x_grid[idx]
    
        plot(x_grid, y, X, posterior_mean, posterior_sig, \
             title='Iteration=%2d,  x_next = %f'%(i+2, x_next))
    
        # 更新
        X = np.atleast_2d([np.r_[X[:, 0], x_next]]).T
        y = np.r_[y, blackbox_func(x_next)]
        
    
    print("Max x=%f" % (x_next))




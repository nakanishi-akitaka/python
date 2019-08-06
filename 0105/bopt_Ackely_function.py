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

Created on Thu Jan  3 17:29:16 2019

@author: Akitaka

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
 
 
def blackbox_func(x):
    """
    Ackely function
    https://qiita.com/tomitomi3/items/d4318bf7afbc1c835dda
    もともとは最小化問題なのでマイナスをかける
    """
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = 20 - 20 * np.exp(-0.2*np.sqrt((x[i,0]**2+x[i,1]**2)*0.5))
        + np.e - np.exp((np.cos(2*np.pi*x[i,0])+np.cos(2*np.pi*x[i,1]))*0.5) 
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
    x_min = -10
    x_max =  10
    n_grid = 100
    x_grid = np.linspace(x_min, x_max, n_grid+1)
    xx1, xx2 = np.meshgrid(x_grid, x_grid)
    xx_grid = np.c_[xx1.ravel(), xx2.ravel()]

    # 初期値として x1=[x_min+1, x_max-1] * x2=[x_min+1, x_max-1] の 4 点の探索をしておく.
    x_grid = np.linspace(x_min+1, x_max-1, 2)
    xx1, xx2 = np.meshgrid(x_grid, x_grid)
    X = np.c_[xx1.ravel(), xx2.ravel()]
    y = blackbox_func(X)
     
    # Gaussian Processs Upper Confidence Bound (GP-UCB)アルゴリズム
    # --> 収束するまで繰り返す(収束条件などチューニングポイント)
    n_iteration = 50
    for i in range(n_iteration):
    
        # 既に分かっている値でガウス過程フィッティング
        # --> カーネル関数やパラメータはデフォルトにしています(チューニングポイント)
        gp = GaussianProcessRegressor()
        gp.fit(X, y)
        
        # 事後分布が求まる
        posterior_mean, posterior_sig = gp.predict(xx_grid, return_std=True)
        
        # 目的関数を最大化する x を次のパラメータとして選択する
        # --> βを大きくすると探索重視(初期は大きくし探索重視しイテレーションに同期して減衰させ
        # 活用を重視させるなど、チューニングポイント)
        idx = acq_ucb(posterior_mean, posterior_sig, beta=100.0)
        x_next = np.atleast_2d(xx_grid[idx])
    
        if(False):
            plot(x_grid, y, X, posterior_mean, posterior_sig, \
                 title='Iteration=%2d,  x_next = %f'%(i+2, x_next))
    
        # 更新
        X = np.r_[X, x_next]
        y = np.r_[y, blackbox_func(x_next)]
        print(i,x_next,blackbox_func(x_next))
    
    print("finished.")
    x_best = np.atleast_2d([0,0])
    print("The theoretically best x, y = ", x_best, blackbox_func(x_best))




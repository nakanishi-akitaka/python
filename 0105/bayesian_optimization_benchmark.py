# -*- coding: utf-8 -*-
"""
ベイズ最適化(GP-UCB アルゴリズム)による探索.
環境
  Python 3
  scikit-learn==0.18.1
  matplotlib==1.5.3
  numpy==1.11.2

ref:
http://www.techscore.com/blog/2016/12/20/機械学習のハイパーパラメータ探索-ベイズ最適/
https://qiita.com/tomitomi3/items/d4318bf7afbc1c835dda

獲得関数の種類について
[1] PI: Probability of improvement (改善確率)
    = x の推定値 y が、y_max + ε より大きくなる確率
      シンプルで分かり易いが、局所解に陥ることも
    = int_ymax^inf 1/sqrt(2 pi σ**2) exp[-(y-μ)**2/(2σ**2)] dy
    = norm.sf(x=ymax, loc=μ, scale=σ)
        t = (μ-y)/σ, dt = -dy/σ
        と変数変換すると、
    =-int_(μ-ymax)/σ^-inf 1/sqrt(2 pi) exp[-t**2/2] dt
    = int_-inf^(μ-ymax)/σ 1/sqrt(2 pi) exp[-t**2/2] dt
    = norm.cdf(x=(μ-ymax)/σ, loc=0, scale=1)

    cdf (Cumulative density function) 累積分布関数
    = [-∞:x] までの積分
    norm.cdf(x=1.0, loc=0, scale=1)
    期待値loc，標準偏差scaleの正規分布の累積分布関数のx=1.0での値を取得します．

    sf (Survival function) 生存関数
    = [x:+∞] までの積分


[2] EI: Expceted Improvement (期待改善量)
    = x の推定値 y が、 y_max よりどれだけ大きいか(改善量)の期待値
      最も一般的に使われている
    = (y-μ) * PI(x) + σ 1/sqrt(2 pi σ**2) exp[-(y-μ)**2/(2σ**2)] 

    ref:
    https://adtech.cyberagent.io/research/archives/24
        EIの公式導出

    PIやEIで、y_max ではなく、 y_max - ξ のように少しずらした値を用いる理由
    https://adtech.cyberagent.io/research/archives/24
        しかし，この方法には注意点があります．それは，改善さえすればPIが良くなってしまうという点です．
        つまり，無限小の改善であってもPIは良くなってしまうため，
        不確実だがもっといいかもしれないところより，
        無限小の改善だけど改善はしそうというところに探索が集まってしまうのです．
        つまり，活用に振りすぎているということになります．
        これを改善しようとして，trade-off parameter ξ≥0を用いる方法が提案されています．


[3] UCB: Upper Confidence Bound (上側信頼限界)
    = 評価値の信頼区間の上限が最も高い点を次に観測
    = μ + sqrt(β)　σ


ref:
20181226ノート

Created on Thu Jan  5 15:13:59 2019

@author: Akitaka


"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from time                             import time


def Rosenbrock_func(x):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = 100*(x[i,1]-x[i,0]**2)**2 + (x[i,0]-1)**2
    return -y 

def Ackely_func(x):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = 20 - 20 * np.exp(-0.2*np.sqrt((x[i,0]**2+x[i,1]**2)*0.5))
        + np.e - np.exp((np.cos(2*np.pi*x[i,0])+np.cos(2*np.pi*x[i,1]))*0.5) 
    return -y 

def Sphere_func(x):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = x[i,0]**2 + x[i,1]**2
    return -y 

def Beale_func(x):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = (1.5 - x[i,0] + x[i,0]*x[i,1])**2 \
        + (2.250 - x[i,0] + x[i,0]*x[i,1]**2)**2 \
        + (2.625 - x[i,0] + x[i,0]*x[i,1]**3)**2
    return -y 

def Booth_func(x):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = (x[i,0] + 2*x[i,1] - 7)**2  + (2*x[i,0] + x[i,1] - 5)**2
    return -y 

def Matyas_func(x):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = 0.26 * (x[i,0]**2 + x[i,1]**2) - 0.48 * x[i,0] * x[i,1]
    return -y 

def Levi_func(x):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = np.sin(3*np.pi*x[i,0]) \
            + (1+np.sin(3*np.pi*x[i,1]))*(x[i,0]-1)**2 \
            + (1+np.sin(2*np.pi*x[i,1]))*(x[i,1]-1)**2
    return -y 

#### PI
def acq_pi(mean, std, ymax):
    if(np.any(std==0)):
        std = std + 0.01
    epsilon = 0.01 # relaxation_value
    lamb = (mean - ymax - epsilon)/std
    acq = np.array([norm.cdf(lamb[i]) for i in range(len(lamb))])
    return np.argmax(acq)

#### EI
def acq_ei(mean, std, ymax):
    if(np.any(std==0)):
        std = std + 0.01
    epsilon = 0.01 # relaxation_value
    lamb = (mean - ymax - epsilon)/std
    acq = np.array([(mean[i] - ymax)*norm.cdf(lamb[i]) 
        + std[i]*norm.pdf(lamb[i]) for i in range(len(lamb))])
    return np.argmax(acq)

#### UCB
def acq_ucb(mean, std, beta=3):
    acq = mean + np.sqrt(beta) * std
    return np.argmax(acq)
 
#### MI
def acq_mi(mean, std, gamma):
    delta = 10 ** -6
    alpha = np.log(2 / delta)
    acq = mean + np.sqrt(alpha)*(np.sqrt(std**2 + gamma) - np.sqrt(gamma) )
    gamma = gamma + std **2
    return np.argmax(acq), gamma
 
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

    start = time()

    # 最適化するべき関数
    ifunc=0
    if(ifunc==0):
        blackbox_func = Ackely_func
        x_best = np.atleast_2d([0,0])    
    elif(ifunc==1):
        blackbox_func = Sphere_func
        x_best = np.atleast_2d([0,0])
    elif(ifunc==2):
        blackbox_func = Rosenbrock_func
        x_best = np.atleast_2d([1,1])
    elif(ifunc==3):
        blackbox_func = Beale_func
        x_best = np.atleast_2d([3,0.5])
    elif(ifunc==4):
        blackbox_func = Booth_func
        x_best = np.atleast_2d([1,3])
    elif(ifunc==5):
        blackbox_func = Matyas_func
        x_best = np.atleast_2d([0,0])
    elif(ifunc==6):
        blackbox_func = Levi_func
        x_best = np.atleast_2d([1,1])

    # 探索範囲
    x_min = -10
    x_max =  10
    n_grid = 100
    x_grid = np.linspace(x_min, x_max, n_grid+1)
    xx1, xx2 = np.meshgrid(x_grid, x_grid)
    xx_grid = np.c_[xx1.ravel(), xx2.ravel()]

    # 初期値として x1=[x_min+1, x_max-1] x2=[x_min+1, x_max-1] の 4点の探索をしておく.
    x_grid = np.linspace(x_min+1, x_max-1, 2)
    xx1, xx2 = np.meshgrid(x_grid, x_grid)
    X = np.c_[xx1.ravel(), xx2.ravel()]
    y = blackbox_func(X)
    ymax = max(y)
    
    # MIでのみ使用
    gamma = 0
    n_iteration = 100
    for i in range(n_iteration):
    
        gp = GaussianProcessRegressor()
        gp.fit(X, y)
        
        posterior_mean, posterior_sig = gp.predict(xx_grid, return_std=True)
        
        # 獲得関数
        iacq=3
        if(iacq==0):
            idx = acq_pi(posterior_mean, posterior_sig, ymax)
        elif(iacq==1):
            idx = acq_ei(posterior_mean, posterior_sig, ymax)
        elif(iacq==2):
            idx = acq_ucb(posterior_mean, posterior_sig, beta=100.0)
        elif(iacq==3):
            idx, gamma = acq_mi(posterior_mean, posterior_sig, gamma)
        x_next = np.atleast_2d(xx_grid[idx])
  
        if(False):
            plot(x_grid, y, X, posterior_mean, posterior_sig, \
                 title='Iteration=%2d,  x_next = %f'%(i+2, x_next))

        X = np.r_[X, x_next]
        y = np.r_[y, blackbox_func(x_next)]
        ymax = max(y)
        if(blackbox_func(x_next) == blackbox_func(x_best)):
            break

        print(i,x_next,blackbox_func(x_next), ymax)
    
    print("finished.")
    idx = np.argmax(y)
    x_opt = np.atleast_2d(X[idx])
    print(x_opt, blackbox_func(x_opt))
    print("The theoretically best x, y = ", x_best, blackbox_func(x_best))

    print('{:.2f} seconds '.format(time() - start))



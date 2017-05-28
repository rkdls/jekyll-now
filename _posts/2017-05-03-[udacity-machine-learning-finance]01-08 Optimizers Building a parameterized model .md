---
layout: post
title: "[udacity-machine-learning-finance]01-08 Optimizers Building a parameterized model"
description: "[udacity-machine-learning-finance]01-08 Optimizers Building a parameterized model"
tags: [finance,machine-learning]
---

Optimizer

Optimizer 는 함수,방정식의 가장 최소값을 찾는 방법이다.

예를들어 다음과 같은 함수가 있다고 하자.

	f(x) = (x-1.5)**2 + 0.5

이함수의 최솟값을 구하려면 어떻게해야할까 . ?

저 함수를 좌표평면으로 그려보면 포물선 모양이 나오고(2차함수) x가 1.5일때 y는 0.5라는 최솟값을 구할수가 있다.

코드상으로 최솟값을 구하는 즉, minimize하는 함수를 구현하려면 살짝 복잡하고, 시간도 오래걸리지만 파이썬의 scipy함수를 사용하면 최적화, 시간등을 신경쓸필요없이 바로 구할수가 있다.

```python
import scipy.optimize as spo

def f(x):
    Y = (x - 1.5)**2 + 0.5
    return Y
def run_f():
    Xguess = 2.0
    min_result = spo.minimize(f, Xguess, method='SLSQP', options={'disp': True})    print(min_result.x, min_result.fun)
```

위와같이 사용할수가 있다. 위의 return 값이 최소일때의 x,값, y값을 구할수가있다.


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo


def fit_line(data, error_func):
    """fit a line to given data, using a supplied error function.
    :parameters
    data: 2D array where each row is a point(X0,Y)
    error_func : function that computes the error between a line and observed data

    Returns line that minimizes the error function.
    """
    l = np.float32([0, np.mean(data[:, 1])])

    x_ends = np.float32([-5, 5])
    plt.plot(x_ends, l[0] * x_ends + l[1], 'm--', linewidth=2.0, label="Initial guess")

    result = spo.minimize(error_func, l, args=(data,), method="SLSQP", options={"disp": True})  # display True
    return result.x


def error(line, data):
    """compute error between given line model and observed data
    line: tuple/list/array (C0, C1) where C0 is slope and C1 is Y-intercept
    data : 2D array where each row is a point(X,Y)

    return error as a single real value.
    """

    err = np.sum((data[:, 1] - (line[0] * data[:, 0] + line[1])) ** 2)
    return err


def run_f():
    l_orig = np.float32([4, 2])
    print("Original line: C0={}, C1={}".format(l_orig[0], l_orig[1]))
    Xorig = np.linspace(0, 10, 21)
    Yorig = l_orig[0] * Xorig + l_orig[1]
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original line")

    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label="data points")
    l_fit = fit_line(data, error)
    print('fitted line C0={}, C1={}'.format(l_fit[0], l_fit[1]))

    plt.plot(data[:, 0], l_fit[0] * data[:, 0] + l_fit[1], 'r--', label="fitted data")
    plt.legend(loc="upper right")
    plt.show()


run_f()
```

위코드는 좀더 심화과정으로, 기울기가 4, y축이 2 인 f(x) = 4x+2인 함수를 가지고, 랜덤한 값(data)를 생성해서 fitting line을 생성하는 코드이다.

error함수를 다시 생성하고, 인자를 하나더 추가하였는데 scipy의 minimize함수에 값을 전달해줄때, args=(data,)이런식으로 튜플로 값을 넘겨주면 scipy에서 알아서 최적의 값을 계산해 전달해준다.

data points는 랜덤으로 bell모양으로 설정해서 계산한 값이고, ploting한 데이터중에 initial data는 그냥 y값들의 평균을 계산해서 가운데값을 계산한것이고,  fitted data는 최적화된 라인을 다시 나타낸것이다.

코드를 찬찬히 살펴보면 어떤의미가 되는지 다 알수있다 (마지막쯔음에 plt.legent(loc="upper right")를 설정해주지 않으면 legend가 보이지않는다)

더복잡한 방정식의 경우도 위와같은 코드에서 조금만 변경하면 된다.
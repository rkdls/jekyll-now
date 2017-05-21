---
layout: post
title: "[udacity-machine-learning-finance]01-04 Statistical anlysis of timeseries"
description: "[udacity-machine-learning-finance]01-04 Statistical anlysis of timeseries"
tags: [finance,machine-learning]
---

Mean은 각 밸류값을 전체합으로 나눈값이고,

Median 은 정렬했을때의 가운데 값을 나타낸다.

dates = pd.date_range('2010-01-01', '2012-12-31')

symbols = ['SPY', 'XOM', 'GOOG', 'GLD']

df = get_data(symbols, dates)

df.std() #dataframe의 standard deviation(표준편차)을 구한다

Rolling statistics : window라고 불리는 창을 기준으로 해당하는 값의 평균을 구한다.

moving average = 이동평균선

주식가격이라는게 moving average를 따라서 이동하게되는데,

moving average와 price의 격차가 커질때, 즉, 표준편차가 커질때 sell signal 또는 buy signal임을 감지한다.

이 방법을 주식 그래프로 나타낸게 bollinger bands이다.

daily return : 주식분석에서 중요한 지표중 하나. 그냥 상승,하강퍼센테이지 이다. (오늘가격/어제가격) -1

ex) 오늘가격이 110달러이고, 어제 가격이 100달러면, daily return = (110/100) -1 = 1.1-1 = 10% 즉, 10%상승

사진은 xom과 s&p의 비교그래프이다. daily return이 상당히 비슷하다.

daily return을 계산해보자 !

```python
def compute_daily_returns(df):
	daily_returns = df.copy()
	daily_returns[1:] = (df[1:]/df[:-1].values)-1 #df.values를 붙인 이유는 numpy형식으로 접근하기 위함이다. (numpy는 0이 완전한 0이아님)
	daily_returns.ix[0,:] = 0
return daily_returns
```

위와같은 식으로 할수있는데,

pandas의 함수를 이용하면 다음과같이 할수도있다.
```python
daily_returns = (df / df.shift(1)) - 1
daily_returns.ix[0, :] = 0
```
cumulative return : 시작일을 기준으로 해당날짜 T의 상승률을 나타낸다.
수학적으로 표현하자면, cumret[T] = (price[T]/price[0])-1

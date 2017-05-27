---
layout: post
title: "[udacity-machine-learning-finance]01-07 Sharpe ratio and other portfolio statistics"
description: "[udacity-machine-learning-finance]01-07 Sharpe ratio and other portfolio statistics"
tags: [finance,machine-learning]
---


종목별 Daily portfolio value를 구해보자!

먼저 4가지 종목을 산다고 치자, [google, facebook, gld, IBM] 각종목의 비중은 40%,40%,10%,10% 이다.


1. prices들을 normalization을한다.
```python
normed = prices/prices[0] #무조건 1로 시작하게되어있다. 기준점을 같이 잡고시작하겠다는뜻
```

2. 비중을 곱해준다.
```python
alloced = normed * allocs
```

3. 그다음에 초기값을 곱해준다.
```python
pos_vals = alloced*start_val
```

4. 마지막으로 각 row를 더해준다
```python
port_val = pos_vals.sum(axis=1)   #각 row를 기준으로(하루마다 계산하겠다는뜻) 총합을 구한다. axis=1은 row를 기준으로 더하겠다는 뜻
```


각 포트폴리오를 평가하는 방법이 여러가지가있는데,
1. daily_returns으로 평가하는 방법.
	저 첫째날은 변화량이 없으니깐 0이 되겠고,
	```python
	daily_rets = daily_rets[1:]
	```

2. 마지막 값에서 첫번째 값의 비율로 계산하는방법 (cumulative).
	```python
	cum_ret = (port_val[-1]/port_val[0])-1
	```

3. 평균값 계산
    ```python
	avg_daily_ret = daily_rets.mean()
	```

4. 표준편차 계산
    ```python
	std_daily_ret = daily_rets.std()
	```

5. 마지막으로 shape ratio 방법이있는데 이방법은 약간 복잡하다고한다.

	Rp:portfolio return

	Rf : risk free rate of return

	Op: stddev of portfolio return

    이라고하면, (Rp-Rf)/Op 로 구할수가있다.
    이 값이 리스크와 총 포트폴리오 수익률, 표준편차를 반영한 식이여서 유용하다고 한다.
    나머지는 다 구할수있겠는데, risk free값은 처음 보는데 어떻게 구해야할까 ?

{% capture images %}
	/images/01-07-Sharpe-ratio-and-other-portfolio-statistics1.png
{% endcapture %}
{% include gallery images=images caption="그림-1" cols=1 %}

여러가지가있다. LIBOR이라는 지표값도있고, 3mo T-bill이라는 값도있고, risk free rate가 매일 조금씩 바뀐다고한다.
아니면 간단하게 0%로 두고도 사용한다.

은행에 넣었을때. 수익률이 10%이면, 백만원을 넣으면 십만원을 얻는다고보자.
그때 연평균 수익률이 10%이고, 날마다(영업일)의 수익률을 구하면, 위의 루트 수식과 같은 그림이 나오고 -1을 빼주고 daily_risk로 계산하면 된다.
그럼 약 0에 가까운 값이 나오고, 분모에있는 std를 구할때의 daily_risk값은 0으로 놓고 보고 상수로 취급한다.


{% capture images %}
	/images/01-07-Sharpe-ratio-and-other-portfolio-statistics2.png
{% endcapture %}
{% include gallery images=images caption="그림-2" cols=1 %}

Sharpe는 연간 기준으로 계산하는 것이라서 매일, 매주, 매달 계산을 하는 방법에 따라서 값이 다르게 나온다.
	그래서 연간 계산법으로 바꿔주려면 k라는 상수를 곱해줘야하는데,
	위 예시와 같이 매일 측정한 값을 연간 값으로 바꾸려면 daily k = 루트252를 곱해주어야 한다.

 이렇게 위와같은 4가지 포트폴리오 평가방법이 있다.
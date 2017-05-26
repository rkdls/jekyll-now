---
layout: post
title: "[udacity-machine-learning-finance]01-05 Incomplete data"
description: "[udacity-machine-learning-finance]01-05 Incomplete data"
tags: [finance,machine-learning]
---
pristine data.

pristine(완전 새것같은, 깨끗한)

사람들의 생각 - 모든 데이터가 매분 기록될것이다.
		   - 사라진 데이터가없을것이다.
실제 - 서로같은 주식이라도 값이 다를수도있다 ( 뉴욕거래소 또는 다른곳에서 서로다른 가격으로 거래될수도있다.)
     - 모든 주식들이 거래날에 거래가 안될수도있다. 중간에 상장폐지될수도있다.


{% capture images %}
	/images/udacity-machine-learning-01-05-1.png
{% endcapture %}
{% include gallery images=images caption="그림-1" cols=1 %}
위와같이 중간에 data가 없어지면 어떻게해야할까.
빈틈을 평균값으로 연결하려고하면 안된다.

{% capture images %}
	/images/udacity-machine-learning-01-05-2.png
{% endcapture %}
{% include gallery images=images caption="그림-2" cols=1 %}

다음과 같이,
1. 끊어지기 전 가장 마지막 price의 값이 그대로 이어진다고 생각해야한다.
2. 그리고 첫번째부터 끊어졌다면 가장 최근값을 그대로 연결해야한다.
(1. fill forward 2. fill backward)

이제 코드로 작성해보자.

dataframe에서 제공하는
fillna() 함수를 이용하자.

df_data.fillna(method='ffill', inplace='TRUE') 해당값으로 설정하면 forward fill이 이루어진다.

하지만 앞에값은 filling이 이루어지지 안았는데, 그건 다른방법을 써야한다.


```python
df_data.bfill(method='ffill', inplace=True)
df_data.bfill(method='bfill', inplace=True)
```


위와같이 ffill을 먼저 해준후, bfill을 해줘야지 정상적으로 표시가된다.
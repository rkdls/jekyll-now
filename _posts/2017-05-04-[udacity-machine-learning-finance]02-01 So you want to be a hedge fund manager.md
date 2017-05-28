---
layout: post
title: "[udacity-machine-learning-finance]02-01 So you want to be a hedge fund manager?"
description: "[udacity-machine-learning-finance]02-01 So you want to be a hedge fund manager?"
tags: [finance,machine-learning]
---

Funds의 종류에는 크게 3가지가 있다.

# ETF (Exchange-traded funds)
주식이랑 비슷하다.

Baskets of Stock이라고 할수있다.

투명하다.

# Mutual funds
end of day에 사거나 팔수있다.

분기마다 disclosure된다.

그래서 좀 불투명하다.

# Hedge funds
사고/팔기가 힘들다 (agreement를 받아야함)

투자자들조차도 어떤종목에 어떻게투자했는지 모른다.

불투명하다.

{% capture images %}
	/images/02-01-So-you-want-to-be-a-hedge-fund-manager.png
{% endcapture %}
{% include gallery images=images caption="그림-1" cols=1 %}

보통의 헷지펀드이 infrastructure는 위와 같이 설정한다.

대충 설명하자면, historical price data와 우리가 target으로 설정한 portpolio를 trading 모델에 넣어서 분석을 실행한다.

그리고 order를 하는데, 한번에 많은 수의 주식을 주문을하게되면 시세가 조정될수도있어서 order를 할때 나눠서 해야한다.(헷지펀드니깐 큰돈이 오가면 시세에 영향을 미칠수가있다)

그다음, 실시간으로 live portfolio를 구성하고, trading 알고리즘으로 target portfolio와 matching 을 시키도록 해야한다.

가운데에 있는 trading 알고리즘 모델은 실시간으로 계속 interactive하게 움직여야한다.

자, 그러면 target portfolio를 구성하는 infra또한 알아보자.!


{% capture images %}
	/images/02-01-So-you-want-to-be-a-hedge-fund-manager2.png
{% endcapture %}
{% include gallery images=images caption="그림-2" cols=1 %}

N-day forcast 항목을 구성해야한다. 아마 machine-learning등을 이용해서 구하면 될것같고, current portfolio와 비교해서 target portfolio를 구성하면 된다. 예를들어 forecast에서 apple이 하락 할거란 예측이 구성된다면, portfolio optimizer를 통해서 portfolio의 밸런스를 맞추면 될것이다.

{% capture images %}
	/images/02-01-So-you-want-to-be-a-hedge-fund-manager3.png
{% endcapture %}
{% include gallery images=images caption="그림-3" cols=1 %}

forcast portfolio를 만드는것이 중요한데, 정보와, 데이터등을 feeding시켜서 머신러닝 모델을 만들어서 forcast 를 알맞게 구성하면 될것같다.

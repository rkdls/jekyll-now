---
layout: post
title: "[udacity-machine-learning-finance]02-04 The Capital Assets Pricing Model"
description: "[udacity-machine-learning-finance]02-04 The Capital Assets Pricing Model"
tags: [finance,machine-learning]
---
CAPM
- Capital Assets Pricing Model 이란것이있다. 이 방법은 각 주식의 가치를 평가하는 모델링 기법이다.
	하나의 수식으로 정리될수있는데,


	$$ r_{i}(t) = \beta_{i}r_{m}(t) + \alpha _{i}(t) $$

	i = 각 개별 종목<br/>
	r = captital return

	따라서 $$ r_{m} $$ 값은 마켓의 return값,상승률을 뜻하게된다. <br/>
	베타($$ \beta $$)와 알파($$ \alpha $$)는 전에 배웠던 daily return에서 나온값이고, <br/>
	마켓의 daily return값과 개별종목의 daily return값을 비교해서 기울기는 b, y축 값은 a로 한다. <br/>
	capm에서는 a값을 0으로 두는경우가 많다.

{% capture images %}
	/images/02-04-The-Capital-Assets-Pricing-Model.png
{% endcapture %}
{% include gallery images=images caption="그림-1" cols=1 %}


passive manager와 active manager
- passive
    - S&P500같은 index펀드를 사고 기다리는것.
- active
    - 스스로 살종목을 정하고 비율을 정함.어떤것은 overweight로 가져가고 어떤것은 underweight로 정한다. (또는 -로 정하기도함)

__CAPM__ : $$ r_{i}(t) = \beta_{i}r_{m}(t) + \alpha _{i}(t)$$ <br/>
식에서 $$ \alpha $$값을 정의할때 <br/>
__passive manager__ 들은 랜덤 또는 0으로 설정하고 거래를 하지만, <br/>
__active manager__ 들은 머신러닝이나, 자신만의 알고리즘을 $$ \alpha $$ 을 정의해서 거래를한다.

{% capture images %}
	/images/02-04-The-Capital-Assets-Pricing-Model2.png
{% endcapture %}
{% include gallery images=images caption="그림-2" cols=1 %}


포트폴리오를 구성할때, 각 개별종목의 capm을 구하고, 내 포트폴리오에서 차지하는 비율($$w_{i}$$)를 곱하고, <br/>
가지고있는 모든 종목을 다더하면 내포트폴리오의 capm이 나온다. <br/>
하지만 식을 간단하게 밑의 식처럼 치환하면 passive manager들과 active manager들이 a값에 대처하는 자세에따라서 식이 바뀐다. (passive들은 a값을 랜덤으로두거나 0으로둔다..)

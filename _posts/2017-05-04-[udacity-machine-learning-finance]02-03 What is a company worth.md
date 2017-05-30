---
layout: post
title: "[udacity-machine-learning-finance]02-03 What is a company worth?"
description: "[udacity-machine-learning-finance]02-03 What is a company worth?"
tags: [finance,machine-learning]
---

주식 가격중에는 true가치 라는것이 있다(진실된 가치 ? )

실제 가격과 true가치를 비교하여서 실제 가격이 true 가치보다 더 높다면 팔때이고, 실제가격이 true가치보다 낮다면 살떄라는 것이다.

이 true가치라는 것을 어떻게 구해야 할까 ?

이것또한 여러가지 방법이 있는데,

1) intrinsic value (본질적인 가치)
	미래의 배당금으로 평가되는 가치라고도한다 모든기업은 아닌데 대부분의 기업이 일정 기간에 배당금을 payback하게 된다. 그 가치에따라서 true value를 결정하게된다.

2) Book value
	회사가 가지고있는 자산에 대한 평가방법 (sum of the assets)

3) Market capitalization
	회사주식의 가치와 주식수의 가치가 얼마나 뛰어난지에 대한 평가 방법이다.(시장이 그기업에대해 어떻게 생각하는지)



The value of a future dollar

미래 가치에 대해서 얘기해 보자.

U.S정부에 돈을 빌려주는대신 1%의 적금을 받는다고 해보자.

미래에(1년후) 1달러의 가치가 현재는 얼마가 될까 ?

$$PV = \frac{FV}{1+IR^i}$$

위 수식과 같이 나오게 되는데, 여기서 i는 얼마나 흐르는지를 나타내는 연도가 되고, IR은 interested rate 즉 금리가 된다.

금리를 수식으로 나타내게되면 0.01이 되므로 위 공식에 맞춰 대입해보면,

현재 가치 (P.V) 는

$$ \frac{1}{1+0.1}($) = 0.99($) $$

즉 0.99$가 된다.

다시말해서 현재 0.99달러를 US정부에 빌려주고 1%의 이자를 받게된다고 치면 1년후에는 1달러를 받게된다는 의미이다.

이 금리가 너무 낮다고 생각되면 일반 사금융 (강좌에서는 Balch Bond라는 교수님 이름을 썻다) 에서 5%라는 금리를 준다고 사람들을 유혹해서 정부보다는 신뢰성이 더 낮지만, 금리가 더높다는 장점으로 투자자들을 유혹한다.


{% capture images %}
	/images/02-03-What-is-a-company-worth.png
{% endcapture %}
{% include gallery images=images caption="그림-1" cols=1 %}

위 그림에 대한 잠시 설명을 하자면, discount rate = 5% 라는 수치는 사용자가 임의의 방법으로 정학 수치 이다.

1년마다 5%의 수익률은 가져다 준다고 하고, (위험성이 높을수록 수익률이 더높다)

그래프는 시간이 지날수록 현재 1달러의 가치는 지수함수적으로 점점더 줄어든다고 본다. (1년후의 PV, 2년후의 PV가 점점 더떨어짐)


$$ \sum_{1}^\infty \frac{FV}{n^i} = \frac{FV}{n-1} $$


위와 같이 정리할수있게되어서 $$\frac{1}{0.05} = 20 $$
즉 Intrinsic value 는 20$ 가 된다.


다음 Book value 에 대해 알아보자

"Total assets minus intagible assets and liabilities"

전체 자산에서 무형의 자산 과 부채를 뺀 가치 라고 할수있겠다.

예를들어 회사에서 공장 $40 짜리가있고, 특허권이 $15, 부채가 $10가 있다면

전체자산(40+15)에서 무형의자산 (15)과 부채(10)를 빼면, 회사의 가치는 $30가 된다.


마지막으로 market capitalization 이 있다.

간단하게 __모든주식수 * 가격이__ 되겠다.

## Summary

Intrinsic value

- 기업이 투자자들에게 주식수를 기준으로 배당을할때 얼만큼의 가치를 가지는지

Book value
- 기업의 자산을 기준으로 나눈 가치

Market capitalization
- 시장에서의 회사의 가치

많은 투자자들이 Intrinsic value와 Market captialization의 편차를 보고 투자를 하기도 한다.

만약 intrinsic value가 하락하면, 주식가격은 높다고 생각되면 주식가격을 줄이는 방법도있고,

배당금이 올라가고 Market capitalization이 낮다면 주식을 사야할 기회라고 생각한다.

또한 시장가격이 Book value보다 낮아지지 않는다고 보는데, Book value이하로 주식 가치가 떨어지면 공격적인 투자자들이 해당주식을 모두 사드려서 Book value를 기준으로 기업을 와해시키고 수익을 얻을수도있기때문이다.
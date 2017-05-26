---
layout: post
title: "[udacity-machine-learning-finance]01-06 Histograms and scatter plots"
description: "[udacity-machine-learning-finance]01-06 Histograms and scatter plots"
tags: [finance,machine-learning]
---
daily-return 은 전날짜와 비교해서 상승률,하강률을 나타내는 그래프이고,

histogram은 각 수익률의 빈도수를 바(bar)그래프로 나타낸것이라고 볼수있다.



{% capture images %}
	/images/01-06-Histograms-and-scatter-plots.png
{% endcapture %}
{% include gallery images=images caption="그림-1" cols=1 %}

Standard deviation은 평균값을 기준으로 얼마나 멀리 떨어진 정도를 나타낸값이고,

Kurtosis는 그리스에서 온단어로 그래프가 얼마나 뾰족한지, 완만한지 정도를 나타내는 지표이다.

Kurtosis가 양수값이면, 양쪽끝의 tail값이 일반적인 가우시안 모양보다 좀더 큰것이고(fat tails), 음수이면 양쪽의 값이 일반적인 가우시안모양보다 좀더 낮다는걸의미한다.(skinny tails)

dataframe에서 histogram을 나타낼때 인자로 bins를 준다.
해당값은 bins의 값으로 나누어서 내림을 하겠다는 의미이다. 그래서 bins=10 이라면, 총10단계의 값이 나온다.


{% capture images %}
	/images/01-06-Histograms-and-scatter-plots2.png
{% endcapture %}
{% include gallery images=images caption="그림-2" cols=1 %}

위 그림은 FB과 SPY의 histogram을 비교한 것이다.
위그림의 의미는 그래프가 더 넓은것(FB)이 daily-return값의 변화량이 훨씬더 많다는것을 의미한다. 아무래도 spy보다는 개별주식인 FB의 주식의 변화량이 더많다고 생각하면 정확하게 나온값이다.
kurtosis의 값도 FB의 값이 더크게나온다(좀더완만하다는 뜻인데 값의 차이가 얼마없다. 0.05정도)

코드는 다음과같다.

```python
daily_returns['SPY'].hist(bins=20, label="SPY")
daily_returns['FB'].hist(bins=20, label="FB")
plt.show()
```

spy를 레퍼런스로 두고 개별종목과 daily return의 비교값을 그래프로 그려보면 그래프가 완전히 똑같을순없다.
그래서 그 차이 값이 크다면 시장과 엇갈린 흐름으로 이동한다고 볼수있고, 차이값이 적다면 시장과 비슷하게 움직인다고 볼수있다.

이 차이를 그래프로 scatter그래프로 나타낼수가 있다.

{% capture images %}
	/images/01-06-Histograms-and-scatter-plots3.png
{% endcapture %}
{% include gallery images=images caption="그림-3" cols=1 %}


위와같이 scatter들의 중심을 연결해서 선을만들면 기울기를 구할수가있는데, 일반적으로 선의 기울기가 1이면, 시장이 1%움직일때 해당종목도 1%움직였다는 의미이고 기울기가 2이면 2배로 움직였다고볼수잇다. 해당 선의 기울기를 Beta값이라고한다.
또다른 요소로는 alpha가 있는데, 해당값은 x가 0일때의 y값으로, a가 크다면 spy보다 xyz라는 개별종목이 더 움직임이 활발하다고 볼수있다.

하지만 기울기가 1이라고 해서 연관성이 크다는 것은 아니다. (slope != correlationship)
연관성은 위의 점들이 선에 얼마나 가까이있냐로 결정할수가있다. 그러므로 기울기가 가파르더라도 점들이 선 근처에 모여있다고하면 좀더 연관성이 높은것으로 판단할수가있다.

correlated를 0~1로 표현할수있는데 0은 연관성이 적고 1은 많다고 생각하면 된다. 결론은 기울기가 크다고해서 연관성이 크다는 뜻은 아니라는거다.

이제 코딩으로 풀어써보자,
numpy의 polyfit이란 함수를 써서 square error함수를 구해보자.

```python
beta_SPY, alpha_GLD = np.polyfit(daily_returns['SPY'], daily_returns['GLD'], deg=1)
plt.plot(daily_returns['SPY'], beta_SPY * daily_returns['SPY'] + alpha_GLD, '-', color='r')
```

위와같이 polyfit함수에 비교할 두값을 주고, degree=1차원으로 설정하면 베타값과 알파값이 반환된다.

plot의 인자로, x값,y값을 주고, 그래프를 그리면 예쁜 1차원 라인이 그려지게된다.


{% capture images %}
	/images/01-06-Histograms-and-scatter-plots4.png
{% endcapture %}
{% include gallery images=images caption="그림-4" cols=1 %}


위그림과 같이 그려지게되고, correlation 을 간편하게 구하기 위해서 pandas에서 제공하는 함수를 구해보자 (1에 가까울수록 유사성이 많다)

```python
daily_returns.corr(method='pearson')
```

위와같이 corr함수를 호출하고, method에 pearson이라는 인자를 주게되면 그림과같은 output이 나오게된다.


마지막으로 2008년에 default가 났던 모기지론에 대해설명하면서,

각 모기지론의 가격이 독립적이지않고, Kurtosis 의 값의 비교를 제대로 안해서 대규모의 경제공황이 왔다고 설명하고있다.


---
layout: post
title: python에서 async url 요청
description: "python에서 async url 요청 파이썬"
tags: [python]
---
python 3.5부터는

함수 정의문 앞에 async 라는 단어를 추가해서 함수를 비동기적으로 처리할수있게됐다. async란 말그대로 asynchrounous 의 약자로 비동기적으로 함수를 이용하겠다는 의미가 된다.

single thread에서 multi thread에서 하던일을 한다는것이 아니라 single thread에서 I/O작업이 진행중이면 해당작업을 이벤트 루프에 등록해두고 다른 작업을 처리한후 I/O작업이 끝나면 그 이후의 작업을 진행하는 방식이다.

예를들어 동기 프로그래밍에서는 코드를 한줄한줄 진행하다가 InputData의 입력을 기다리게 될경우가 있는데,  그동안 thread는 다음 구문을 진행하지 않고,blocking이 되어서 진행을 못하게된다.

대기하고있는 thread는 그만큼의 resource의 낭비가 되며 이것을 잘활용할수있다면 python을 좀더 유용하게 프로그래밍 할수있게된다.

비동기(non-blocking) 처리를 이용하면 Thread가 blocking되어서 대기하고있는동안에 cpu의 연산작업을 시킬수가있다.

이작업을 하려면 프로그래밍 언어 level에서 해당 처리 지원해야하며 그래서 최근에 파이썬에 나온것이 async/await 문법이다.

```python
import asyncio

async def compute(x, y): #5
    print("Compute %s + %s ..." % (x, y))
    await asyncio.sleep(1.0)   #6
    return x + y  #7
async def print_sum(x, y):  #3
    result = await compute(x, y) #4
    print("%s + %s = %s" % (x, y, result))  #8

loop = asyncio.get_event_loop() #1
loop.run_until_complete(print_sum(1, 2)) #2
loop.close()
```
결과값 : 1 + 2 = 3

[https://docs.python.org/3/library/asyncio-task.html](https://docs.python.org/3/library/asyncio-task.html)

1)  코루틴 스케쥴링을 시작한다.

2)  future객체를 받는데, 해당 인수가 coroutine object일경우 ensure_future()메소드로  future객체로 감싼다. 객체를 받아서 해단 코루틴이 완료될때까지 대기한다. 완료되면 Future의 결과를 return한다.

3) 코루틴을 사용하겠다는 async 문장을 추가한 함수를 선언한다.

4) 코루틴 함수내에서 또다른 코루틴을 호출해서 해당 루틴이 끝날때까지 대기한다.

5) compute 함수내로 진입하게된다.

6) 여기서도 await문을 만나게되어서 해당 코루틴(asyncio.sleep(1.0)) 이 끝날때까지 대기하게된다.

7) 결과값을 return한다.

8) return 된 값으로 print를 한다.

{% capture images %}
	/images/python_ascync.png
{% endcapture %}
{% include gallery images=images caption="async flow" cols=1 %}


위 코드의 흐름도.
Task객체는 run_until_complete()가 호출될때 내부적 생성된다.

위의 결과만봤을때는 await를 썼을대와 안썼을때의 흐름의 차이가 없지만 await의 사용법을 보여주고있다.

```python
Blocking IO

import requests
import time
from bs4 import BeautifulSoup

urls = ['http://google.com', 'http://naver.com', 'http://daum.net', 'http://depromeet.com', 'http://facebook.com',
        'http://dreamtamercom.wordpress.com']


def get_request(url):
    return requests.get(url).text


def get_html():
    for i, url in enumerate(urls):
        http = get_request(url)
        bs = BeautifulSoup(http, 'lxml')
        bs.findAll('script')


start_ = time.time()
get_html()
print("총걸린시간 {}".format(time.time() - start_))  # 약 3~4초. 환경에따라 다름
```

코드는 단순하게 url을 요청하고 각요청후에 BeautifulSoup로 html의 모든 script태그의 Object를 가져오는 코드다.

위 코드의 문제점은 request.get(url) 함수를 실행시키면 해당하는 url로 네트워크 요청을 보내게되고, thread는 네트워크요청에 대한 응답을 받을때까지 다음 작업을 wait하게되어서 blocking상태가 되어 resource의 낭비가 일어나게된다.

Non-Blocking IO

async와 await를 사용해서 코드를 개선해보자.

처음에 requests라이브러리만 가지고 코드를 개선시키려고하면 해당라이브러리에서 async처리를 지원해줘야하는데 requests 라이브러리는 coroutine지원이 안된다.

그래서 aiohttp라는 async request를 처리해주는 라이브러리가 필요하다.

```python
pip install aiohttp
```
```python
import asyncio
import time
from aiohttp import ClientSession
from bs4 import BeautifulSoup

urls = ['http://google.com', 'http://naver.com', 'http://daum.net', 'http://depromeet.com', 'http://facebook.com',
        'http://dreamtamercom.wordpress.com']


async def url_request(url):
    async with ClientSession() as session:  # 6
        async with session.get(url) as response:  # 7
            html = await response.read()  # 8
            bs = BeautifulSoup(html, 'lxml')  # 9
            bs.findAll('script')


async def get_html():  # 2
    htmls = [url_request(url) for url in urls]  # 3
    for i, completed in enumerate(asyncio.as_completed(htmls)):  # 4
        await completed  # 5


start_ = time.time()
event_loop = asyncio.get_event_loop()
try:
    event_loop.run_until_complete(get_html())  # 1
except:
    event_loop.close()
print("async 총걸린시간 {}".format(time.time() - start_))  # 약 1~2초 환경에따라 다름
```

1) #3 coroutine 객체를 생성하고 리스트에 담는다.

2) #4 asyincio.as_completed(htmls) Future 객체로 감싸져있는 coroutine을 반환한다.

3) #5 반환된 coroutine이 await문을 만나면 해당 함수를 실행한다.

4) #6~7 aiohttp에서는 세션을 열고, 해당 요청이 끝나면 세션도 닫아줘야한다. 그래서 async with을써서 세션이 제대로 닫히게 도와준다. session이 닫히는 작업도 async 작업으로 진행된다.

5) #8 await문을 만나면 이제 aiohttp의 요청을 시작하고, response를 읽어들이다.

6) #9 그후 응답이 온 순서대로 다음 작업을 진행하게 된다. (리스트 내의 순서는 보장되지안음)

위 처리사항중 주의해야할것이, #7 번에서 실질적으로 request요청이 일어나는것이 아니라 await문을 만나면 그순간에 request가 일어난다는 점이다.

## 요약

 requests 라이브러리처럼 아직 파이썬에서 async를 지원하지않는게 더많아서 개인이 만든 function이나 지원하는 라이브러리가 아니면 async를 구현하기가 좀 까다로울수도있다.
프로그래밍 하기가 좀 까다롭지만 blocking과 non-blocking 의 성능차이는 엄청나다.
단순연산작업만 많을경우에는 그다지 효용이 없다.
node-js, nginx등이 non-blocking방식으로 설계되어있다.

### 참고

[https://pawelmhm.github.io/asyncio/python/aiohttp/2016/04/22/asyncio-aiohttp.html](https://pawelmhm.github.io/asyncio/python/aiohttp/2016/04/22/asyncio-aiohttp.html)

[https://docs.python.org/3/library/asyncio-task.html](https://docs.python.org/3/library/asyncio-task.html)

[http://soooprmx.com/wp/archives/6882](http://soooprmx.com/wp/archives/6882)


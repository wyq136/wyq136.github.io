title: 函数的导数
date: 2018-09-20 21:09:46
description: "整理一些常见的函数的导数"
categories: Math
tags:
  - Math
---

## 常见基本函数导数

|导数名|原函数|导函数|
|-|-|-|
|常函数（常数）|$y=C$ (C为常数)|$y^\prime=0$|
|幂函数|$y=x^n$|$y^\prime=nx^{n-1}$|
|指数函数|$y=a^x$|$y^\prime=a^x\ln x$|
||$y=e^x$|$y^\prime=e^x$|
|对数函数|$y=\log_a x$|$y^\prime=\frac{1}{x\ln a}$|
||$y=\ln x$|$y^\prime=\frac{1}{x}$|
|正弦函数|$y=\sin x$|$y^\prime=\cos x$
|余弦函数|$y=\cos x$|$y^\prime=-\sin x$

## 复合函数求导

原函数：$y^\prime=f(g(x))$， 其中 $y=f(u)$， $u=g(x)$

使用链式法则求导：$y^\prime = f^\prime(u)u^\prime(x) = f^\prime(g(x))g^\prime(x)$

## 导数的四则运算

$$(u \pm v)^\prime = u^\prime \pm v^\prime$$

$$(uv)^\prime = u^\prime v + u v^\prime$$

$$(\frac{u}{v})^\prime = \frac{u^\prime v - u v^\prime}{v^2}$$

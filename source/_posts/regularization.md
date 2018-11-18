
title: 正则化（regularization）
description: ' '
date: 2018-11-17 22:30:33
categories: 
tags:
  - Machine Learning
  - Deep Learning
---

## 什么是正则化？

正则化主要的作用是防止模型过拟合，其原理是对网络中的参数进行惩罚（约束），防止网络模型中的参数过大而过于偏向某一个特征。常见的正则化有L1和L2正则化。

## L1正则化

对模型进行正则化一般是将正则项直接加到损失函数后面，L1正则化是把网络中所有的参数的绝对值相加。

$$loss_{regularization} = loss + \lambda \sum_{j=1}^{n} |\theta_j|$$

其中 $\lambda$ 为正则化系数，$n$ 为参数个数。

## L2正则化

$$loss_{regularization} = loss + \frac{\lambda}{2} \sum_{j=1}^{n} \theta_j^2$$

其中 $\lambda$ 为正则化系数（这里除于2是为了求导时计算简便），$n$ 为参数个数。

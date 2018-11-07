title: 逻辑回归（Logistic Regression）
description: ' '
date: 2018-11-03 21:06:26
categories:
tags:
  - Machine Learning
  - Deep Learning
---

## 模型

$$
\begin{aligned}
h_\theta(x) =& {g(\bf{\theta{^T} x}}) \\\\
g(z) =& \frac{1}{1 + e^{-z}}
\end{aligned}
$$

$g(z)$ 称为sigmoid函数，函数图像为：

![sigmoid image](/resource/images/sigmoid.png)

逻辑回归实际上是在线性回归的基础上在加上一个sigmoid函数（非线性变换），合并起来就是：

$$
h_\theta(x) = \frac{1}{1 + e^{-\bf{\theta{^T}x}}}
$$

## 损失函数

$$
\begin{aligned}
loss =  J(\theta) = & \frac{1}{m} \sum_{i=1}^{m}Cost(h_\theta(x^i) - y^i) \\\\
= & -  \frac{1}{m} [\sum_{i=1}^{m} y^i \log{h_\theta(x^i)} + (1- y^i) \log{(1 - h_\theta(x^i))}]
\end{aligned}
$$

逻辑回归输出的值在 0-1 之间，使用log损失，当标签 $y^i=1$ 时，预测结果 $h_\theta(x^i)$ 也为 1 时损失为0，预测结果与 1 相差越多，损失越大。当标签 $y^i=0$ 时同理。

## 优化目标

求使得损失函数最小的参数 $\bf{\theta}$ 。

$$\min_{\theta} J(\theta)$$

## 梯度下降

求损失函数关于每一个参数 $\theta_j$ 的梯度并不断迭代。

$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m}{(h_\theta(x^i) - y^i)x_j^i}$$

其中，$x^i$ 表示第i条数据，$x_j$ 表示第j个特征，对应的参数为 $\theta_j$； 这个式子和线性回归的完全一致！

## 具体求导过程

## 代码实现

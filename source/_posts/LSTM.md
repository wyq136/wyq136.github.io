title: LSTM
description:
date: 2019-01-19 15:24:28
categories:
tags:
---

## 简介

LSTM全称是 Long Short Term Memory Network（长短时记忆网络），它也是一种循环神经网络（RNN）算法。

<!-- more -->

在普通的 RNN 中，经常会出现以下两个问题：

- 梯度爆炸：梯度太大导致程序出错
- 梯度消失：原始RNN无法处理长距离依赖

梯度爆炸相对来说比较好解决，比如：可以设置一个梯度阈值，当梯度超过这个阈值的时候可以直接截取。而相对于梯度消失来说，这个会比较难解决一些。

LSTM 算法的出现就是为了解决梯度消失的问题。

## 模型

为了解决 RNN 模型中长距离依赖梯度消失的问题，LSTM 中引入了一个新的记忆单元，以及三个门：输入门、遗忘门和输出门。

### 前向计算

![lstm cell](/resource/images/lstm-cell.png)

$$
\begin{aligned}
i_t =& \sigma(W_{ii}x_t + W_{hi}h_{t-1} + b_i) \\\\
f_t =& \sigma(W_{if}x_t + W_{hf}h_{t-1} + b_f) \\\\
o_t =& \sigma(W_{io}x_t + W_{ho}h_{t-1} + b_o) \\\\
g_t =& \tanh(W_{ig}x_t + W_{hg}h_{t-1} + b_g) \\\\
c_t =& f_t \times c_{t-1} + i_t \times g_t \\\\
h_t =& o_t \times \tanh c_t \\\\
\end{aligned}
$$

在上图和公式中，$i_t, f_t, o_t$ 分别为输入、遗忘、输出门； $g_t$ （图中为$c^\prime_i$）是当前时刻前馈计算的结果； $c_t$ 是长期记忆单元， $h_t$ 是这一个时刻 LSTM 网络的输出结果。$\sigma$ 是 sigmoid 激活函数，$\times$ 是指向量的对应值相乘。

### 梯度计算

LSTM 的梯度计算和 RNN 的差不多，只是多了一些参数复杂一些。我们可以使用梯度检查的方法来检验计算的梯度是否正确。

## 代码实现

[完整代码](https://github.com/hf136/models/tree/master/LSTM)

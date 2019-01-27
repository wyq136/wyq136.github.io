title: Attention——深度学习中的注意力机制
description: 
date: 2019-01-27 11:24:28
categories:
tags:
---

## 什么是 Attention ？

我们来一起看着下面这张图片，并且读一下下面这句话。

> 一只黄色的小猫带着一个鹿角帽子趴在沙发上。

![cat](/resource/images/cat.png)

在读这句话的过程中，你的注意力是不是会发生变化？我相信大多数人是这样的：当读到“小猫”的时候，注意力在猫身上；当读到“鹿角帽子”的时候，注意力在鹿角帽子上。

这就是人类的注意力，它是会随着时间发生变化的。

<!-- more -->

## 神经网络中的 Attention 机制

### seq2seq 模型

![seq2seq](/resource/images/seq2seq.png)

seq2seq 是指 sequence to sequence，这类模型的输入是一个序列 x1、x2、x3 ...，输出也是一个序列 y1、y2 ...。它通常由两个类似于 LSTM 的循环神经网络（RNN）构成，也常被称作 Encoder-Decoder 模型，第一个 RNN 进行 encode，第二个 RNN 作为 decode。

一般的 Encoder-Decoder 模型，Encoder阶段可以表示为：

$$
\begin{aligned}
    h_t =& f(x_t, h_{t-1}) \\\\
    c   =& q(h_1, ... ,h_T) \\\\
\end{aligned}
$$

其中： $h_t$ 是 n 维实数向量，表示编码阶段RNN的 t 时刻的隐藏状态；c 是各个时刻隐藏状态生成的向量；𝑓 和 𝑞 是非线性函数。例如：$f$ 可以是 LSTM，$q$ 函数取最后一个时刻的输出结果，$q({h_1, ... ,h_{T_x}}) = h_T$ 。


Decoder 阶段可以表示为：

$$
\begin{aligned}
    y_t =& g(y_{t-1}, s_t, c) \\\\
\end{aligned}
$$

其中， $y_t$ 是 t 时刻的输出结果，初始化 $y_0$ 为零向量，$s_t$ 为 t 时刻隐藏层状态，c 为 Encoder 阶段的输出结果；g 是非线性函数。例如：g 是一个 LSTM 单元，把 $y_{t-1}， c$ 拼起来当做输入。

在机器翻译、对话生成等场景中经常会用到这类模型，但这类模型是有一些局限性的。

- 局限性：编码和解码之间的唯一联系就是一个固定长度的语义向量 C ；在输入序列较长的情况下信息损失更加严重。

### Attention 模型原理

由于单纯的seq2seq会存在一些问题，于是人类便发明在神经网络加入模拟人类注意力的机制。

下面我们为上面的 Encoder-Decoder 模型加入一种 Attention 机制。

Attention 机制改变的是 Encoder-Decoder 模型中的 Decoder 阶段；Encoder 阶段不变，将 Decoder 阶段变成：

$$
\begin{aligned}
    y_t =& g(y_{t-1}, s_t, c_t)
\end{aligned}
$$

单一向量 c 变成了一组向量 $c_t$ ，它的计算方式为：

$$
c_t = \sum_j^T{\alpha_{tj} h_j}
$$

其中， $h_j \in {(h_1, ... ,h_T)}$ ，即 $h_j$ 为Encoder阶段j时刻的输出，$\alpha_{tj}$ 为t时刻第j个隐藏层的权重系数，它可以通过一个前馈神经网络学习得到：

$$
\begin{aligned}
    c_{tj} = \frac{\rm{exp}(e_{tj})} {\sum_{k=1}^T{ \rm{exp}(e_{tk}) }} \\\\
    e_{tj} = a(s_{t-1}, h_j)
\end{aligned}
$$

其中，𝑎 是一个前馈神经网络，比如：𝑎 可以是一个由 tanh 作为激活函数的神经元。$s_{t-1}$ 为t-1时刻的decoder阶段的隐藏层状态，$h_j$ encoder 阶段 j 时刻的输出。

总的来说就是：

> t 时刻的向量 $c_t$ 是由encoder阶段各个时刻的输出 $h_t$ 加权得到的结果；
> 而这个加权的权重是由decoder阶段t-1时刻的隐藏层状态 $s_{t-1}$ 和encoder阶段各个时刻的输出 $h_t$ 通过一个前馈神经网络并归一化之后的结果。

### Attention 机制可视化

![attention model](/resource/images/attention-1.png)

图中，横轴为：输入的英文单词序列，纵轴为：输出的法语单词序列；每一行是权值 𝛼 组成的向量；越亮的地方权重越大。

这相当于实现了一种 “软对齐” 机制，所以注意力机制有时也叫做 “对齐模型” （Alignment Model）。

## 参考资料

https://arxiv.org/abs/1409.0473
https://mp.weixin.qq.com/s/_Ru6GMcrSO25bTs8vM6FmA


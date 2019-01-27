title: 循环神经网络（Recurrent Neural Network）
description: 
date: 2018-12-28 21:40:59
categories:
tags:
---

## 简介

在自然语言处理（NLP）中，需要处理的数据通常都是不定长的。例如，我们要构建一个神经网络模型，将下面这两句话翻译成英文：
- 这一世诺言从不曾忘。
- 深度学习的概念源于人工神经网络的研究。

这两句话的长度是不一样的，一般的神经网络输入的特征纬度是固定的，显然不能很好的解决这个问题，于是便出现了循环神经网络（Recurrent Neural Network，RNN）。

## 模型

### 基本的循环神经网络

一个最基本的循环神经网络由输入层，隐藏层和输出层构成，如下图所示：

![rnn-1](/resource/images/rnn-1.jpg)

<!-- more -->

这里的x、s、o分别是输入层、隐藏层和输出层，它们都是一个向量；W、U、V是连接层与层之间的权重矩阵。
RNN和一般的神经网络最大不同在于：
> RNN多了一个从隐藏层到隐藏层($s => s$)的过程，使RNN拥有了“记忆”的功能。
> (注意：这里的s要把他它看层多个隐藏层的多个神经单元，s是隐藏层单元构成的向量)

在RNN网络中，我们需要引入一个时间（顺序）的概念，我们把上图展开，RNN可以画成这样：

![rnn-2](/resource/images/rnn-2.jpg)

从图中可以看到，t 时刻的RNN网络输入值是 $x_t$，输出值是 $o_t$，隐藏层的值是 $s_t$，它的值取决于输入值 $x_t$ 和 t-1 时刻的隐藏层的值 $s_{t-1}$。

### 前向计算

RNN的每时间步的计算过程如下：
$$
\begin{aligned}
s_t =& g(Ux_t + Ws_{t-1} + b) \\\\
o_t =& f(Vs_t)
\end{aligned}
$$

其中，g、f 是激活函数，s 为隐藏层的值， b 是偏差项；隐藏层 s 的初始值 $s_0$ 为零向量。

可以看出，RNN网络最后输出的结果受到所有输入序列 $x_1, x_2 ... x_T$ 的影响。因为隐藏 $s_{t-1}$ 保存了前面 t-1 个 x 值的结果，隐藏层 s 充当了一个“记忆”的角色。

### 优化目标
同样是求使得损失函数最小的权重 U 、 W 、V 、b；损失函数的形式根据具体的任务会有所不同。

### 梯度计算

这里RNN使用到的计算梯度的算法是BPTT（Back Propagation Trough Time），加上了时间的概念，是一种基于时间的反向传播算法。

虽然名字听上去很高大上的样子，但其实并不复杂，和普通的反向传播算法也差不多，把RNN展开之后，一样可以使用链式求导，这其实就很简单了。

![RNN backward](/resource/images/rnn-3.png)

如上图所示，把 RNN 展开之后，RNN 每一时刻的反向传播求导过程和普通的神经网络是一样。根据任务的不同，有可能每一个时刻都有误差传递，也可能只有最后一个时刻有误差传递。

展开后的 RNN 可以看成共享权重的全连接神经网络模型，只要使用链式求导分别求出每一个时刻的权重梯度，最后再把所有时刻的梯度相加求和就可以得到最终的 RNN 权重梯度。

## 梯度爆炸和梯度消失

在序列很长的时候，RNN 模型训练过程中，很容易出现梯度爆炸（梯度很大）或者梯度消失（梯度几乎为0）的问题，导致模型无法正常拟合。

这是为什么呢？

链式求导求解梯度的过程其实一个连乘的过程： 
$$
\frac{\partial{S_n}}{\partial{S_{n-1}}} \frac{\partial{S_{n-1}}}{\partial{S_{n-2}}} ... \frac{\partial{S_2}}{\partial{S_1}}
$$
当序列很长的时候，如果每个阶段梯度都大于1的话，梯度就会爆炸，比如: $10^9$ ；如果每个阶段梯度都小于1的话，梯度就会消失，比如: $0.1^9$ ；

对于梯度消失，其实指的是长距离的梯度消失，即长距离依赖会消失，训练时梯度不能在较长序列中一直传递下去，从而使RNN无法捕捉到长距离的影响。也就是说 RNN 的“记忆力”有限，在处理较长的序列时，往往会“忘记”序列前面的内容。由于整个模型的梯度是各个时刻梯度之和，所以整个模型的梯度还不会消失。

## 代码实现

这里实现了一个简单的 RNN 模型，其中激活函数使用的是 Relu 激活函数。[完整代码](https://github.com/hf136/models/tree/master/RNN)

一个 RNN 时间步的计算过程，其实就和普通的神经网络是一致的。

``` python
class RNNCell:
    """
    一个 RNN 时间步的计算过程
    """
    def __init__(self, in_size, hidden_size):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.w_i2h = np.random.normal(0, 0.1, (in_size, hidden_size))
        self.w_h2h = np.random.normal(0, 0.1, (hidden_size, hidden_size))
        self.bias = np.random.normal(0, 0.1, (1, hidden_size))

    def relu(self, x):
        x[x < 0] = 0
        return x

    def forward(self, x, h):
        self.i2h = x.dot(self.w_i2h)
        self.h2h = h.dot(self.w_h2h)
        self.h_relu = self.relu(self.i2h + self.h2h + self.bias)
        return self.h_relu

    def backward(self, grad, i, h):
        if i.ndim == 1:
            i = np.expand_dims(i, axis=0)
        if h.ndim == 1:
            h = np.expand_dims(h, axis=0)

        self.grad_h_relu = grad
        self.grad_h = self.grad_h_relu.copy()
        self.grad_h[h < 0] = 0
        self.grad_w_h2h = h.T.dot(self.grad_h)
        self.grad_w_i2h = i.T.dot(self.grad_h)
        self.grad_bias = self.grad_h
        self.grad_h_in = self.grad_h.dot(self.w_h2h.T)

        return self.grad_h_in

    def update_weight(self, lr):
        self.w_i2h -= lr * self.grad_w_i2h
        self.w_h2h -= lr * self.grad_w_h2h
        self.bias -= lr * self.grad_bias

```

完整的 RNN 序列计算过程。

``` python
class RNN:
    """
    完整的 RNN 序列计算过程
    """
    def __init__(self, in_size, hidden_size):
        self.h_state = []
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.rnncell = RNNCell(in_size, hidden_size)

    def forward(self, x):
        self.h_state = []
        self.x = x
        h = np.zeros(self.hidden_size)
        for i in x:
            self.h_state.append(h)
            h = self.rnncell.forward(i, h)
        self.h_out = h
        return self.h_out

    def backward(self, grad):
        self.grad_w_i2h = np.zeros((self.in_size, self.hidden_size))
        self.grad_w_h2h = np.zeros((self.hidden_size, self.hidden_size))
        self.grad_bias = np.zeros((1, self.hidden_size))

        for i in range(len(self.h_state) - 1, -1, -1):
            x = self.x[i]
            h = self.h_state[i]
            grad = self.rnncell.backward(grad, x, h)
            self.grad_w_i2h += self.rnncell.grad_w_i2h
            self.grad_w_h2h += self.rnncell.grad_w_h2h
            self.grad_bias += self.rnncell.grad_bias
        return grad

    def update_weight(self, lr):
        self.rnncell.w_i2h -= lr * self.grad_w_i2h
        self.rnncell.w_h2h -= lr * self.grad_w_h2h
        self.rnncell.bias -= lr * self.grad_bias
        return self.rnncell.w_i2h, self.rnncell.w_h2h, self.rnncell.bias

```

## 参考资料

https://zybuluo.com/hanbingtao/note/541458

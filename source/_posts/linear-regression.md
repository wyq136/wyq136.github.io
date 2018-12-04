title: 线性回归（Linear Regression）
description: ' '
date: 2018-09-27 22:30:33
categories: Machine Learning
tags:
  - Machine Learning
  - Deep Learning
---

## 模型

$$y={\bf{wx}} +b$$

其中，${\bf{w}}$ 和 ${\bf{x}}$ 都是向量， ${\bf{w}} = w_1, w_2, ..., w_n$ 表示要学习的模型参数, ${\bf{x}} = x_1, x_2, ..., x_n$ 表示模型的输入。

## 损失函数（代价函数）

$$L({\bf{w}}, b) = \frac{1}{2m} \sum_{i=1}^{m}{(y^\prime_i - y_i)^2}$$

其中，$m$ 表示训练样本数， $y^\prime$ 表示模型输出结果， $y$ 表示实际结果。

## 优化目标

求使得损失函数最小的参数 ${\bf{w}}$ 和 $b$ 。

$$\min_{w,b}L({\bf{w}}, b)$$

## 梯度下降

对损失函数求每个 $w_j$ 和 $b$ 的偏导数，并通过下式不断迭代得到较优的参数：

$$w_j := w_j - \alpha \frac{1}{m} \sum_{i=1}^{m}{(y^\prime_i - y_i)x_j}$$

$$b := b - \alpha \frac{1}{m} \sum_{i=1}^{m}{(y^\prime_i - y_i)}$$

其中， $\alpha$ 为学习速率， $\alpha$ 后面的项为偏导数。

## 代码实现

[完整代码](https://github.com/hf136/models/tree/master/LinearRegression)

``` python
# 定义参数 w 和 b
w = random.random()
b = 0

learning_rate = 1e-4
for epoch in range(1000):
    # 定义模型，前向计算
    pred_y = w * x + b

    # loss
    loss = 0.5 * np.square(pred_y - y).sum() / y.size
    print('epoch {}, loss {}'.format(epoch, loss))

    # 计算梯度（求导）
    grad_w = ((pred_y - y) * x).sum() / y.size
    grad_b = (pred_y - y).sum() / y.size

    # 更新参数
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
```

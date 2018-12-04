
title: 人工神经网络（Artificial Neural Network）
description: 
date: 2018-11-18 10:30:33
categories: 
tags:
  - Machine Learning
  - Deep Learning
---

## 模型

一个三层的神经网络模型如下：

![nn model](/resource/images/nn.png)

<!--more-->

神经网络可分为输入层、隐藏层和输出层，输入层一般除了输入的特征数据之外，还会有一个偏差项（bias）。神经网络一般会包含一个或者多个隐藏层，隐藏层一般由多个神经元（Neural Unit）构成，上图中的有一个隐藏层，隐藏层中有4个神经元。输出层根据具体不同的任务可以由神经元或者普通的线性回归等构成。

### 神经元

神经元一般由一个线性回归和一个激活函数（非线性部分）构成，例如：逻辑回归可以作为神经网络中的一种神经元。

常见的激活函数有：sigmoid函数、Relu、tanh等。

### 前向计算

设输入层、隐藏层和输出层的单元个数分别为 n，l， k，则3层的神经网络一共有 $n \cdot l + l \cdot k$ 个参数。

$$
\begin{aligned}
z_2 =& \Theta_1 X \\\\
a_2 =& g(z_2) \\\\
z_3 =& \Theta_2 a_2 \\\\
a_3 =& g(z_3) \\\\
\hat{y} =& a_3 \\\\
\end{aligned}
$$

其中，X 是输入的特征向量，$\hat{y}$ 是神经网络输出的结果，$\Theta_1$ 是一个 $n \cdot l$ 的参数矩阵（输入层为n，隐藏层为l），$\Theta_2$ 是隐藏层到输出层的参数矩阵，大小为 $l \cdot k$，$g(z)$ 为激活函数，这里使用sigmoid函数作为激活函数。这里的 $a, z$ 都是向量，函数 $g(z)$ 也是指对向量中的每一个元素做非线性变换。

## 损失函数

神经网络一般使用交叉熵，即使用和逻辑回归类似的损失函数，输出层的每一个输出单元是一个逻辑回归损失，并且求和。

$$
\begin{aligned}
loss =  J(\theta) = & \frac{1}{m} \sum_{i=1}^{m} \sum_{i=1}^{K} Cost(\hat{y}^i_k - y^i_k) \\\\
= & -  \frac{1}{m} \sum_{i=1}^{m} \sum_{i=1}^{K} [y^i_k \log{\hat{y}^i_k} + (1- y^i_k) \log{(1 - \hat{y}^i_k)}]
\end{aligned}
$$

其中，K 输出层输出单元个数， m 为训练样本数，$\hat{y}^i_k$ 为第i个样本的第k个输出单元的输出结果。

## 优化目标

求使得损失函数最小的参数 ${\Theta}$ 。

$$\min_{\Theta} J(\Theta)$$

## 计算梯度

使用链式求导法则计算梯度，$\Theta_2$ 的梯度为：

$$
\frac{\partial J}{\partial\Theta_2} = \frac{\partial J}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z_3}   \frac{\partial z_3}{\partial \Theta_2} = (-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}) (\hat{y}(1-\hat{y})) \cdot a_2 = (\hat{y} - y) \cdot a_2
$$

$\Theta_1$ 的梯度为：

$$
\frac{\partial J}{\partial\Theta_1} = \frac{\partial J}{\partial \hat{y}}   \frac{\partial \hat{y}}{\partial z_3}   \frac{\partial z_3}{\partial a_2}    \frac{\partial a_2}{\partial z_2}  \frac{\partial z_2}{\partial \Theta_1} = \{(\hat{y} - y) \Theta_2 [a_2(1-a_2)]\} \cdot X
$$

这实际上也就是反向传播（Backpropagation）算法。令 $\delta^L = \hat{y} - y$ ，则 $\delta^{l-1} = \delta^l \Theta_{l-1} [a_{l-1}(1-a_{l-1})]$ ，梯度 $\Delta^l = \delta^{l+1} a_l$ 。

**注意**：上面两个式子中的 $\Theta， a，z， \hat{y}$ 指的是矩阵或者向量中的元素分别求偏导数，上面这么写是为了简便。求导过程写成矩阵运算的形式也更方便，需注意进行相应的转置变换。

随着模型结构越来越复杂，每次都去计算梯度比较复杂，现在已经有很多深度学习框架可以进行**自动微分**计算梯度。下面我们还是使用链式求导法则来自己计算一下神经网络的梯度。

## 代码实现
[完整代码](https://github.com/hf136/models/blob/master/ArtificialNeuralNetwork/raw_neural_network.py)

``` python
input_size = 2
hidden_size = 5
output_size = 1

# 生成参数theta
theta1 = np.random.rand(input_size + 1, hidden_size)
theta2 = np.random.rand(hidden_size, output_size)

# 添加偏差项
ones = np.ones((X.shape[0], 1))
X = np.concatenate((X, ones), axis=1)

learning_rate = 1e-1
for epoch in range(10001):
    # 定义模型，前向计算（这里的隐藏层没有添加偏差项，也可以在每层隐藏层都加上偏差项）
    z2 = X.dot(theta1)
    a2 = 1 / (1 + np.exp(-z2))
    z3 = a2.dot(theta2)
    pred_y = 1 / (1 + np.exp(-z3))

    # loss
    loss = 0.5 * np.square(pred_y - y).sum() / y.size
    print('epoch {}, loss {}'.format(epoch, loss))

    # 输出训练时的准确率
    if epoch % 100 == 0:
        pred_label = pred_y >= 0.5
        true_label = y >= 0.5
        diff = pred_label == true_label
        accuracy = diff.mean()
        print('accuracy {}'.format(accuracy))

    # 使用链式求导计算梯度（和反向传播是一致的）
    grad_z3 = pred_y - y
    grad_theta2 = grad_z3.T.dot(a2).T
    grad_a2 = grad_z3.dot(theta2.T)
    grad_z2 = grad_a2 * a2 * (1 - a2)
    grad_theta1 = X.T.dot(grad_z2)

    # 更新参数
    theta2 -= learning_rate * grad_theta2
    theta1 -= learning_rate * grad_theta1
```

预测结果：

![lr res](https://github.com/hf136/models/raw/master/docs/images/ann_res.png)
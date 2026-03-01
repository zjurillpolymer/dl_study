import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import multinomial
import d2l


# C=torch.randn(3,4) # 符合高斯分布（0，1）
# print(C)

# X=torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# print(X)


### 按元素计算

# x = torch.tensor([1.0, 2, 4, 8])
# y = torch.tensor([2, 2, 2, 2])
# print(x+y)
# print(x**y)

# X=torch.arange(12,dtype=torch.float).reshape((3,4))
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# # 按第一个维度即行维度连接两个矩阵
# Z=torch.cat((X,Y),dim=0)
# # print(Z)
# print(X==Y)


# a=torch.arange(3).reshape((3,1))
# print(a)
# b = torch.arange(2).reshape((1, 2))
#
# print(b)
# # 将a复制列，b复制行，然后按元素相加
# print(a+b)
#
# print(a-b)

# X=torch.arange(12).reshape(3,4)
# Y=torch.zeros(3,4)
# print(X)
# print(Y)
# print(X==Y)
# print(X!=Y)
# print(X>Y)
# print(X<Y)
# # 切片操作与range相同
# print(X[-1])
# print(X[1:10])
#
# X[1,2]=9
# print(X)

# X[0:2,:]=12
# print(X)

# A=X.numpy()
# print(type(A))
# a = torch.arange(3).reshape((3, 1))
# b = torch.arange(2).reshape((1, 2))
# print(a)
# print(b)
# print(a+b)
#
#
# x = torch.tensor([1., 2., 3.])
# y = x + 10
# print(y)

# features = torch.randn(2, 4)
# print(features)
# bias=torch.tensor([1.,2.,3.,4.])
# print(bias+features)

# X=torch.arange(24,dtype=torch.float).reshape((2,3,4))
# Y=torch.arange(12,dtype=torch.float).reshape((3,4))
# print(X)
# print(Y)
# print(X+Y)

#
# images = torch.randn(8, 3, 32, 32)
# print(images)

# 简单的线性代数

# x=torch.arange(12).reshape(3,4)
# y=torch.arange(12).reshape(4,3)
#
# print(x*y.T)

# x=torch.arange(12).reshape(2,3,2)
# print(len(x)) 即x.shape[0]
# print(x.size())
# print(x.shape)
# print(x.numel())

# A = torch.arange(20).reshape(5, 4)
# print(A)
# print(A.T)

# X=torch.arange(24).reshape(6,4).float() ##必须输入浮点数类型才能求均值
# print(X)
# print(X.sum(axis=0))
# print(X.mean(axis=0))
# print(X.mean(axis=0).shape)
# A=X.mean(axis=0,keepdims=True)
# print(A)
# print(A.shape)
# print(X/A)

# m*n的矩阵必须乘以n维的向量
# x=torch.tensor([1.0,2.0,3.0,4.0,5.0,6.0])
# # y=torch.tensor([1,2,3,4,5,6])
# # print(torch.dot(x,y))
# # print(torch.mv(A,x))
# B=torch.arange(12).reshape(6,2)
# print(torch.mm(A,B))
# print(A@B)
# A=torch.arange(24).reshape(2,3,4)
#
# print(A)
# print(A.sum(axis=0))
# print(A.sum(axis=1))
# print(A.sum(axis=2))


# 创建一个 2×3×4 的张量
# # 创建一个 2×3×4 的张量
# X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
# print("X.shape:", X.shape)  # torch.Size([2, 3, 4])
#
# # 默认 norm：全局 L2 范数
# norm_all = torch.linalg.norm(X)
# print("torch.linalg.norm(X):", norm_all)  # 标量
#
# # 手动验证：展平后求 L2 范数
# manual = torch.sqrt(torch.sum(X ** 2))
# print("Manual L2 norm:", manual)
#
# # 两者相等？
# print("Equal?", torch.allclose(norm_all, manual))  # True

# 定义函数f(x)
# def f(x):
#     return x**2 - 3*x + 2
#
# # 计算f(x)在x=2处的值
# x0 = 2
# y0 = f(x0)
#
# # 导数函数f'(x)
# def df(x):
#     return 2*x - 3
#
# # 在x=2处的导数值（斜率）
# slope = df(x0)

# 切线方程
# def tangent_line(x, x0, y0, slope):
#     return slope * (x - x0) + y0
#
# # 准备数据
# x = np.linspace(-1, 5, 400)
# y = f(x)
# tangent_y = tangent_line(x, x0, y0, slope)
#
# # 创建图形和坐标轴
# plt.figure(figsize=(8, 6))
# plt.plot(x, y, label='f(x) = $x^2 - 3x + 2$')
#
# plt.plot(x, tangent_y, label='Tangent at x=2', linestyle='--', color='red')  # 绘制切线
#
# # 添加标题和坐标轴标签
# plt.title('Function and its Tangent Line')
# plt.xlabel('x')
# plt.ylabel('y')
#
# # 显示图例
# plt.legend()
#
# # 显示图像
# plt.grid(True)
# plt.axhline(0, color='black',linewidth=1)
# plt.axvline(0, color='black',linewidth=1)
# plt.show()


# x=torch.arange(4.0)
# x.requires_grad_(True)
# # print(x.grad)
# # y=2*torch.dot(x,x)  #y 是一个标量（0 维张量），这是 .backward() 的前提（只能对标量调用）。
# # print(y)
# # y.backward()
# # print(x.grad)
# # print(x.grad==4*x)
# x.grad.zero_()
# print(x.grad)
# y=x*x
# y.sum().backward()
# print(x.grad)


# x = torch.arange(4.0, requires_grad=True)  # [0., 1., 2., 3.] 追踪其梯度
# # y = x*x  # y = [0., 1., 4., 9.] → 非标量！
# #
# # print("y:", y)  # tensor([0., 1., 4., 9.], grad_fn=<PowBackward0>)
# #
# #
# #
# # u = y.detach() #复制这个数，但把它变成常量
# # print("u:", u)
# # z = u * x
# # z.sum().backward()
# # print(x.grad == u)


# def f(a):
#     b = a * 2
#     while b.norm() < 1000:
#         b = b * 2
#         if b.sum() > 0:
#             c = b
#         else:
#             c = 100 * b
#     return c
#
# a = torch.randn(3, requires_grad=True)
# print(a)
# d = f(a)
# d.sum().backward()
# print(a.grad==d/a)


fair_probs = torch.ones([6]) / 6
# print(fair_probs)
# print(multinomial.Multinomial(100000, fair_probs).sample()/100000)  ##生成一个形如fair_probs的向量，并随机取样

# counts = multinomial.Multinomial(10, fair_probs).sample((500,))
# cum_counts = counts.cumsum(dim=0)
# estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
# d2l.set_figsize((6, 4.5))
# for i in range(6):
# d2l.plt.plot(estimates[:, i].numpy(),
# label=("P(die=" + str(i + 1) + ")"))
# d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
# d2l.plt.gca().set_xlabel('Groups of experiments')
# d2l.plt.gca().set_ylabel('Estimated probability')
# d2l.plt.legend();



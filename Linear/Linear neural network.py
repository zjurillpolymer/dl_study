import math
import time
import numpy as np
import torch
from d2l import torch as d2l
import random
from torch import nn
from torch.utils import data

class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()



# n = 10000
# a = np.ones([n])
# b = np.ones([n])
# c = np.zeros(n)
# timer = Timer()
# for i in range(n):
#     c[i] = a[i] + b[i]
# print(f'{timer.stop():.5f} sec')


# def normal(x, mu, sigma):
#     p = 1 / math.sqrt(2 * math.pi * sigma**2)
#     return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
# # 再次使用numpy进行可视化
# x = np.arange(-7, 7, 0.01)
# # Mean and standard deviation pairs
# params = [(0, 1), (0, 2), (3, 1)]
# d2l.plot(x.asnumpy(), [normal(x, mu, sigma).asnumpy() for mu, sigma in params], xlabel='x',
#          ylabel='p(x)', figsize=(4.5, 2.5),
#          legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])

# ### 生成数据集
# true_w=torch.tensor([2,-3.4])
# true_b=4.2
# features,labels=d2l.synthetic_data(true_w,true_b,1000)
#
# ### 读取数据集
# def load_array(data_arrays,batch_size,is_train=True):
#     ### is_train：布尔值，指示是否用于训练。如果是训练，则打乱数据顺序（shuffle）；否则不打乱（保持原有顺序，如用于测试或验证）。
#     dataset=data.TensorDataset(*data_arrays)
#     return data.DataLoader(dataset,batch_size,shuffle=is_train)
#
# batch_size=10
# data_iter=load_array((features,labels),batch_size)
# # print(next(iter(data_iter)))
#
# ### 定义模型
# net=nn.Sequential(nn.Linear(2,1))
# # for param in net.parameters():
# #     print(param.shape)  ##net
#
# ### 初始化模型参数
# net[0].weight.data.normal_(0,0.01)
# net[0].bias.data.fill_(0) # net[0]是指第0层，既linear层本身
#
# ### 定义损失函数
# loss=nn.MSELoss()
#
# ### 定义优化算法
# trainer = torch.optim.SGD(net.parameters(),lr=0.03)
#
# ### 训练
# num_epochs=3
# for epoch in range(num_epochs):
#     for X,y in data_iter:
#         l= loss(net(X),y) ##loss是损失函数loss=nn.MSELoss()，net(X)相当于是y_predicted
#         trainer.zero_grad() #清空之前的梯度
#         l.backward() #自动微分
#         trainer.step() #更新参数
#     l=loss(net(features),labels)
#     print(l)
#     print(net[0].weight.data)
#     print(net[0].bias.data)




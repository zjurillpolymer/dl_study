import os
# 必须在导入任何可能使用 OpenMP 的库之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
from attention_visualization import show_heatmaps

n_train=50
x_train,_=torch.sort(torch.rand(n_train)*5) #(50,)
# print(x_train.shape)

def f(x):
    return 2*torch.sin(x)+x*0.08

y_train=f(x_train)+torch.normal(0,0.5,(n_train,)) ##真实值+正态分布的噪音
# print(y_train.shape)
x_test=torch.arange(0,5,0.1) ## 训练输入
y_truth=f(x_test) ## 真实值
n_test=len(x_test) ## 测试样本数

def plot_kernel_reg(y_hat):
    d2l.plot(x_test,[y_truth,y_hat],'x','y',figsize=(5,5),
             xlim=[0,5],ylim=[-1,5])
    d2l.plt.plot(x_train,y_train,'o',alpha=0.5)
    plt.show()

# plot_kernel_reg(y_hat=torch.zeros(n_test))
#
# y_hat_new=torch.repeat_interleave(y_train.mean(),n_test) 将y_train.mean()重复n_test次
# plot_kernel_reg(y_hat_new)
#
# X_repeat = x_test.repeat_interleave(n_train,dim=0).reshape((-1, n_train))
# attention_weights=nn.functional.softmax(-(X_repeat-x_train)**2/2,dim=1)
# y_hat=torch.matmul(attention_weights,y_train)
# plot_kernel_reg(y_hat)
# show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
#               xlabel='sorted training inputs',
#               ylabel='sorted testing inputs')





## 小批量矩阵乘法
# X=torch.ones((2,1,4))
# Y=torch.ones((2,4,6))
# print(torch.bmm(X,Y).shape)


class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.randn((1,), requires_grad=True))

    def forward(self, queries, keys, values):

        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        #(50,49)
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w) ** 2 / 2, dim=1  ##两个向量直接相减乘以权重
        )
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)


'''
queries：每一行是同一个样本值重复 49 次（比如第 0 行全是 x1，第 1 行全是 x2）；
keys：每一行是当前样本之外的 49 个不同样本值（比如第 0 行是 x2,x3,...,x50，第 1 行是 x1,x3,...,x50）；
'''

''''
repeat(n)	整体重复	[a,b,c].repeat(2) → [a,b,c,a,b,c]	按整个张量重复
repeat_interleave(n)	元素级重复	[a,b,c].repeat_interleave(2) → [a,a,b,b,c,c]	按单个元素重复
'''

X_tile = x_train.repeat((n_train, 1)) #(2500,1)
Y_tile = y_train.repeat(n_train, 1) #(2500,1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1)) #(50*50)的对角矩阵的取反
#(50,49)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])
for epoch in range(10):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    #x_train就是queries
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
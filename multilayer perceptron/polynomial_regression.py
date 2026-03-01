import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

### 初始化数据集
max_degree=20
n_train,n_test=100,100
true_w=np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features=np.random.normal(size=(n_train+n_test,1)) ### x 一个200*1的张量 服从正态分布
np.random.shuffle(features)
poly_features=np.power(features,np.arange(max_degree).reshape(1,-1)) # 广播，200*20的矩阵
# for i in range(max_degree):
#     poly_features[:,i]/=math.gamma(i+1) #除以阶乘，归一化函数，得到e^x的泰勒展开
labels=np.dot(poly_features,true_w)
labels+=np.random.normal(scale=0.01,size=labels.shape)

true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]

### 训练和调试模型
def evaluate_loss(net,data_iter,loss):
    metric=d2l.Accumulator(2)
    for X,y in data_iter:
        out=net(X)
        y=y.reshape(out.shape)
        l=loss(out,y)
        metric.add(l.sum(),l.numel())
    return metric[0]/metric[1]



def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss()  # 注意：这里可以直接用默认的 'mean'，因为我们自己控制训练循环
    input_shape = train_features.shape[-1]

    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_features.shape[0])

    # 使用 DataLoader 或 d2l.load_array（只要能迭代 (X, y) 就行）
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        net.train()  # 设置为训练模式（虽然这里没 dropout/bn，但好习惯）
        for X, y in train_iter:
            y_pred = net(X)
            l = loss(y_pred, y)  # 默认 reduction='mean'，返回标量
            trainer.zero_grad()
            l.backward()
            trainer.step()

        # 每 100 轮打印一次权重（避免刷屏）
        if (epoch + 1) % 100 == 0:
            w = net[0].weight.data.numpy().flatten()
            print(f'epoch {epoch + 1}, weight: {w}')

    return net

print(# 从多项式特征中选取所有维度
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
)
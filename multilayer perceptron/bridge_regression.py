# import torch
# from d2l import torch as d2l
#
# n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
# true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
# train_data = d2l.synthetic_data(true_w, true_b, n_train)
# train_iter = d2l.load_array(train_data, batch_size)
# test_data = d2l.synthetic_data(true_w, true_b, n_test)
# test_iter = d2l.load_array(test_data, batch_size, is_train=False)
#
#
#
#
# ### 初始化模型参数
# def init_params():
#     w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
#     b = torch.zeros(1, requires_grad=True)
#     return [w, b]
#
# ### 定义L2范数惩罚
# def l2_penalty(w):
#     return torch.sum(w.pow(2)) / 2
#
#
# ### 定义训练代码
# def train(lambd):
#     w,b=init_params()
#     net=lambda X:d2l.linreg(X,w,b)
#     loss=d2l.squared_loss
#     num_epochs,lr=100,0.03
#
#     for epoch in range(num_epochs):
#         for X, y in train_iter:
#             l=loss(net(X),y)+lambd*l2_penalty(w)
#             l.sum().backward()
#             d2l.sgd([w,b],lr,batch_size)
#
#         if (epoch + 1) % 5 == 0:
#             train_loss = d2l.evaluate_loss(net, train_iter, loss)
#             test_loss = d2l.evaluate_loss(net, test_iter, loss)
#             print(f'epoch {epoch + 1}, train loss {train_loss:.6f}, test loss {test_loss:.6f}')
#
#             # 最终输出 w 的 L2 范数
#     print('w的L2范数是：', torch.norm(w).item())
#
#
# train(lambd=3)


import torch
from torch import nn
from d2l import torch as d2l


def train_concise(wd):
    # 初始化数据
    n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    train_data = d2l.synthetic_data(true_w, true_b, n_train)
    train_iter = d2l.load_array(train_data, batch_size)
    test_data = d2l.synthetic_data(true_w, true_b, n_test)
    test_iter = d2l.load_array(test_data, batch_size, is_train=False)

    # 1. 定义模型：线性层
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()  # 初始化权重

    # 2. 定义损失函数
    loss = nn.MSELoss(reduction='none')

    # 3. 定义优化器：关键点在于 weight_decay 参数
    # wd 对应你代码中的 lambd，会自动在梯度更新时加入 L2 惩罚
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd},
        {"params": net[0].bias}  # 偏差通常不加正则化
    ], lr=0.003)

    # 训练循环
    for epoch in range(100):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()

    print('w的L2范数是：', net[0].weight.norm().item())


# 调用
train_concise(0)  # 无正则化
train_concise(3)  # 有正则化
train_concise(5)
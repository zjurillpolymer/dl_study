# # import torch
# # import torchvision
# # from sympy import partition
# # from torch.utils import data
# # from torchvision import transforms
# # from d2l import torch as d2l
# # import matplotlib.pyplot as plt
# #
# # d2l.use_svg_display()
# # class Accumulator:  #@save
# #     """在n个变量上累加"""
# #     def __init__(self, n):
# #         self.data = [0.0] * n
# #
# #     def add(self, *args):
# #         self.data = [a + float(b) for a, b in zip(self.data, args)]
# #
# #     def reset(self):
# #         self.data = [0.0] * len(self.data)
# #
# #     def __getitem__(self, idx):
# #         return self.data[idx]
# #
# #
# #
# # class Animator:
# #     """在动画中绘制数据（适配 PyCharm / 脚本环境）"""
# #     def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
# #                  ylim=None, xscale='linear', yscale='linear',
# #                  fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
# #                  figsize=(3.5, 2.5)):
# #         if legend is None:
# #             legend = []
# #         self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
# #         if nrows * ncols == 1:
# #             self.axes = [self.axes]
# #         self.config_axes = lambda: self._set_axes(
# #             self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
# #         self.X, self.Y, self.fmts = None, None, fmts
# #         plt.ion()  # 启用交互模式
# #         plt.show()
# #
# #     def _set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
# #         ax.set_xlabel(xlabel)
# #         ax.set_ylabel(ylabel)
# #         ax.set_xscale(xscale)
# #         ax.set_yscale(yscale)
# #         if xlim:
# #             ax.set_xlim(xlim)
# #         if ylim:
# #             ax.set_ylim(ylim)
# #         if legend:
# #             ax.legend(legend)
# #
# #     def add(self, x, y):
# #         if not hasattr(y, "__len__"):
# #             y = [y]
# #         n = len(y)
# #         if not hasattr(x, "__len__"):
# #             x = [x] * n
# #         if not self.X:
# #             self.X = [[] for _ in range(n)]
# #         if not self.Y:
# #             self.Y = [[] for _ in range(n)]
# #         for i, (a, b) in enumerate(zip(x, y)):
# #             if a is not None and b is not None:
# #                 self.X[i].append(a)
# #                 self.Y[i].append(b)
# #         self.axes[0].cla()
# #         for x_data, y_data, fmt in zip(self.X, self.Y, self.fmts):
# #             self.axes[0].plot(x_data, y_data, fmt)
# #         self.config_axes()
# #         self.fig.canvas.draw()
# #         self.fig.canvas.flush_events()
# #         plt.pause(0.01)  # 短暂暂停以刷新图像
# #
# #
# # # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# # # 并除以255使得所有像素的数值均在0～1之间
# # # ### 读取数据集
# # # trans = transforms.ToTensor()
# # # mnist_train = torchvision.datasets.FashionMNIST(
# # #     root="../data", train=True, transform=trans, download=True)
# # # mnist_test = torchvision.datasets.FashionMNIST(
# # #     root="../data", train=False, transform=trans, download=True)
# # # # print(len(mnist_train), len(mnist_test))
# # # # print(mnist_train[0][0].shape)
# # # def get_fashion_mnist_labels(labels):  #@save
# # #     """返回Fashion-MNIST数据集的文本标签"""
# # #     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
# # #                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
# # #     return [text_labels[int(i)] for i in labels]
# # #
# # # def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
# # #     """绘制图像列表"""
# # #     figsize = (num_cols * scale, num_rows * scale)
# # #     _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
# # #     axes = axes.flatten()
# # #     for i, (ax, img) in enumerate(zip(axes, imgs)):
# # #         if torch.is_tensor(img):
# # #             # 图片张量
# # #             ax.imshow(img.numpy())
# # #         else:
# # #             # PIL图片
# # #             ax.imshow(img)
# # #         ax.axes.get_xaxis().set_visible(False)
# # #         ax.axes.get_yaxis().set_visible(False)
# # #         if titles:
# # #             ax.set_title(titles[i])
# # #     return axes
# # # X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# # #
# # # ### 读取小批量
# # #
# # # batch_size=256
# # #
# # def get_dataloader_workers():
# #     return 4
# # # train_iter=data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers())
# # # timer=d2l.Timer()
# # # for X, y in train_iter:
# # #     continue
# # # # print(f'{timer.stop():.2f} sec')
# #
# #
# # ### 整合所有组件，集成获取数据集+小批量读取的功能
# # def load_data_fashion_mnist(batch_size,resize=None):
# #     trans=[transforms.ToTensor()] # 把原始数据（比如一张图片）转换成模型能用的格式（比如一个张量）
# #     if resize:
# #         trans.insert(0,transforms.Resize(resize))
# #     trans=transforms.Compose(trans)
# #     mnist_train = torchvision.datasets.FashionMNIST(
# #         root="../data", train=True, transform=trans, download=True)
# #     mnist_test = torchvision.datasets.FashionMNIST(
# #         root="../data", train=False, transform=trans, download=True)
# #     return (data.DataLoader(mnist_train, batch_size, shuffle=True,
# #                             num_workers=get_dataloader_workers()),
# #             data.DataLoader(mnist_test, batch_size, shuffle=False,
# #                             num_workers=get_dataloader_workers()))
# #
# #
# # train_iter, test_iter = load_data_fashion_mnist(256, resize=64)
# # # for X, y in train_iter:
# # #     print(X.shape, X.dtype, y.shape, y.dtype)
# # #     break
# #
# # ### 初始化模型参数
# # num_inputs = 4096
# # num_outputs = 10
# #
# # W=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
# # b=torch.zeros(num_outputs,requires_grad=True)
# #
# #
# # ### 定义softmax操作
# # # X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# # # print(X.sum(0, keepdim=True))
# # # print(X.sum(1, keepdim=True))
# #
# # def softmax(X):
# #     X_exp = torch.exp(X)
# #     partition=X_exp.sum(1,keepdim=True)
# #     return torch.div(X_exp,partition)
# #
# # ### 定义模型
# # def net(X):
# #     return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
# #
# #
# #
# #
# # ### 定义损失函数
# #
# # y = torch.tensor([0, 2])
# # y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# # # print(y_hat[[0, 1], y])
# #
# #
# # def cross_entropy(y_hat, y):
# #     return - torch.log(y_hat[range(len(y_hat)), y])  ### y_hat中存储的是每一个样本属于某类的概率，y储存的是真实结果，y_hat[range(len(y_hat)), y]相当于是从y_hat中提取出真实结果对应的概率值
# #
# # # print(cross_entropy(y_hat, y))
# #
# # ### 分类精度
# #
# # def accuracy(y_hat, y):
# #     if len(y_hat.shape) >1 and y_hat.shape[1]>1:
# #         y_hat=y_hat.argmax(axis=1)  ### 第一个维度，即行维度，每行是一个样本
# #     cmp=y_hat.type(y.dtype)==y
# #     return float(cmp.type(y.dtype).sum())
# #
# # # print(accuracy(y_hat, y)/len(y))
# #
# # def evaluate_accuracy(net, data_iter):  #@save
# #     """计算在指定数据集上模型的精度"""
# #     if isinstance(net, torch.nn.Module):
# #         net.eval()  # 将模型设置为评估模式
# #     metric = Accumulator(2)  # 正确预测数、预测总数
# #     with torch.no_grad():
# #         for X, y in data_iter:
# #             metric.add(accuracy(net(X), y), y.numel())
# #     return metric[0] / metric[1]
# #
# # # print(evaluate_accuracy(net,test_iter ))
# #
# #
# # ### 训练
# # def train_epoch_ch3(net, train_iter, loss, updater):
# #     if isinstance(net, torch.nn.Module):
# #         net.train()
# #     metric = Accumulator(3)
# #     for X, y in train_iter:
# #         y_hat = net(X)
# #         l = loss(y_hat, y)
# #         if isinstance(updater, torch.optim.Optimizer):
# #             updater.zero_grad()
# #             l.mean().backward()
# #             updater.step()
# #         else:
# #             l.sum().backward()
# #             updater(X.shape[0])
# #         metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
# #     return metric[0] / metric[2], metric[1] / metric[2]  # ←←← 移到循环外！
# #
# # def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
# #     """训练模型（定义见第3章）"""
# #     animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
# #                         legend=['train loss', 'train acc', 'test acc'])
# #     for epoch in range(num_epochs):
# #         train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
# #         test_acc = evaluate_accuracy(net, test_iter)
# #         animator.add(epoch + 1, train_metrics + (test_acc,))
# #     train_loss, train_acc = train_metrics
# #     # assert train_loss < 0.5, train_loss
# #     # assert train_acc <= 1 and train_acc > 0.7, train_acc
# #     # assert test_acc <= 1 and test_acc > 0.7, test_acc
# #
# #
# # lr = 0.1
# #
# # def updater(batch_size):
# #     return d2l.sgd([W, b], lr, batch_size)
# #
# #
# # num_epochs = 10
# # train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
#
#
#
#
#
#
#
#
#
#
# import torch
# import torchvision
# from torch.utils import data
# from torchvision import transforms
# from d2l import torch as d2l
#
# # 保留你已有的 load_data_fashion_mnist
# def load_data_fashion_mnist(batch_size, resize=None):
#     trans = [transforms.ToTensor()]
#     if resize:
#         trans.insert(0, transforms.Resize(resize))
#     trans = transforms.Compose(trans)
#     mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
#     mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
#     return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
#             data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))
#
#
# import torch.nn as nn
#
# # 输入尺寸：64x64 = 4096
# net = nn.Sequential(
#     nn.Flatten(),           # 将 (B, 1, 64, 64) → (B, 4096)
#     nn.Linear(4096, 10)     # 线性层：输出 logits（不加 softmax！）
# )
#
#
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, std=0.01)
#         nn.init.zeros_(m.bias)
#
# net.apply(init_weights)
# loss = nn.CrossEntropyLoss()  # 内部自动做 log_softmax + NLLLoss
# lr = 0.1
# trainer = torch.optim.SGD(net.parameters(), lr=lr)
#


import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms, datasets

# 数据加载
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    train_dataset = datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    test_dataset = datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4))

# 模型
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(64*64, 10)
)

# 初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_weights)

# 组件
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 数据
train_iter, test_iter = load_data_fashion_mnist(256, resize=64)

# 训练（用 d2l 或自定义）
from d2l import torch as d2l
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs=10, updater=trainer)
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义你的 MLP
# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(20, 256)
#         self.out = nn.Linear(256, 10)
#
#     def forward(self, x):
#         return self.out(F.relu(self.hidden(x)))
#
# # 实例化模型
# model = MLP()
#
# # 创建一个 batch 的输入（batch_size=4, input_dim=20）
x = torch.randn(2, 20)
# target = torch.randint(0, 10, (4,))  # 真实标签：4 个类别（0~9）
#
# # 前向传播
# output = model(x)  # shape: [4, 10]
#
# # 计算损失（CrossEntropyLoss 内部含 softmax）
# criterion = nn.CrossEntropyLoss()
# loss = criterion(output, target)
#
# print("Loss:", loss.item())
#
# # 👇 关键：触发自动微分（反向传播）
# loss.backward()
#
# # 现在，所有参数的 .grad 属性已被填充！
# for name, param in model.named_parameters():
#     print(f"{name}: grad shape = {param.grad.shape}, "
#           f"grad norm = {param.grad.norm().item():.4f}")



# class MySequential(nn.Module):
#     def __init__(self,*args):  ###*args 表示可以传入任意多个参数（比如 2 个、3 个 layer），它们会被打包成一个元组。
#         super().__init__()
#         for idx,module in enumerate(args):
#             self._modules[str(idx)]=module ### self._modules 是 nn.Module 内部的一个 有序字典（OrderedDict），专门用来存储子模块。
#
#     def forward(self, x):
#         for block in self._modules.values():
#             x=block(x)
#         return x
# net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# # net = MySequential(
# #     nn.Linear(20, 256),   # 被存为 _modules["0"]
# #     nn.ReLU(),            # 被存为 _modules["1"]
# #     nn.Linear(256, 10)    # 被存为 _modules["2"]
# # )
# print(net(x))


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)

    # 使⽤创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复⽤全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
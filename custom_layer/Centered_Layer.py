import torch
import torch.nn.functional as F
from torch import nn

class Centered_Layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x-x.mean()

net=nn.Sequential(
    nn.Linear(8,128),
    Centered_Layer()
)

# Y=net(torch.rand(4,8))
# print(Y)
# print(Y.mean())

# class MyLinear(nn.Module):
#     def __init__(self, in_units,units):  ### 输入可学习参数
#         super().__init__()
#         self.weight=nn.Parameter(torch.randn(in_units,units))
#         self.bias=nn.Parameter(torch.randn(units,))
#     def forward(self,x): ### 前向计算，即描述公式的地方
#         linear=torch.mm(x,self.weight)+self.bias
#         return F.relu(linear)
#
#
# linear=MyLinear(5,3)
# print(linear.weight)



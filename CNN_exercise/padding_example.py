import torch
from torch import nn

'''深度学习框架要求的输入格式通常是 (Batch_size, Channels, Height, Width)。

    第一个 1：表示 Batch size（这一批只有 1 张图）。

    第二个 1：表示 Channels（单通道，如灰度图）。'''

def comp_conv2d(conv2d,X):
    X=X.reshape((1,1)+X.shape)
    Y=conv2d(X)
    return Y.squeeze() ### Y.shape[2:]：取 Y 形状的第 3 和第 4 个维度（即真正的 Height 和 Width）。

conv2d=nn.Conv2d(1,1,kernel_size=3,padding=1)
X=torch.rand(size=(8,8))
'''输入的X是8*8的张量，卷积核的维度是3，填充了1，则输出维度为(8-3+1*2)/1+1=4'''
print(comp_conv2d(conv2d,X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))

print(comp_conv2d(conv2d, X).shape)

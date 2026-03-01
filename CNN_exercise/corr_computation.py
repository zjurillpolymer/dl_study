import torch
from torch import nn
from d2l import torch as d2l

### 互相关运算
def corr2d(X,K):
    h,w=K.shape
    Y=torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=(X[i:i+h,j:j+w]*K).sum()
    return Y

# X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# print(corr2d(X,K))


### 卷积层
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(kernel_size))
        self.bias=nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x,self.weight)+self.bias


### 构造⼀个6×8像素的⿊⽩图像。中间四列为⿊⾊（0），其余像素为⽩⾊（1）。
# X=torch.ones((6,8))
# X[:,2:6]=0
# # print(X)
# K=torch.tensor([[1.0,-1.0]])  ### 它是“导数”的离散化 水平差分算子 1*2的张量，因此将其应用于矩阵，第二个维度将会降维，检测水平方向上的亮度突变
# Y=corr2d(X,K)
# # print(Y) ### 维度：n_X-n_K+1 因此输出是一个6*7的矩阵
# # print(corr2d(X.t(),K))
#
# conv2d=nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
# X=X.reshape((1,1,6,8))
# Y=Y.reshape((1,1,6,7))
# lr=3e-2
#
# for i in range(10):
#     Y_hat=conv2d(X)
#     l=(Y-Y_hat)**2
#     conv2d.zero_grad()
#     l.sum().backward()
#     conv2d.weight.data[:]-=lr*conv2d.weight.grad ### 梯度下降
#     if (i+1)%2==0:
#         print(l.sum())
#
# print(conv2d.weight.reshape((1,2)))

# X=torch.ones(8,8)
# for i in range(8):
#     X[i,i]=0
# print(X)
# print(corr2d(X,K))
# print(corr2d(X.t(),K))
# print(corr2d(X,K.t()))


def im2col_conv2d(X,K):
    h,w=X.shape
    kh,kw=K.shape

    out_h=h-kh+1
    out_w=w-kw+1

    rows=[]
    for i in range(out_h):
        for j in range(out_w):
            window=X[i:i+kh,j:j+kw].flatten()
            rows.append(window)
    X_col=torch.stack(rows)
    K_vec=K.flatten().reshape(-1,1)
    Y_linear=torch.matmul(X_col,K_vec)

    return Y_linear.reshape(out_h,out_w)
X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, -1.0],
                  [1.0, -1.0]])

print("输入 X:\n", X)
print("卷积核 K:\n", K)
print("矩阵乘法得到的卷积结果:\n", im2col_conv2d(X, K))
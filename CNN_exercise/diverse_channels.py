import torch
from d2l import torch as d2l

# 假设已经定义了 d2l.corr2d 或类似的单层卷积函数
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

def corr2d_multi_in(X, K):
    # 将输入和核的对应通道相加
    return sum(corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_in_out(X, K):
    # 注意这里必须是 corr2d_multi_in(X, k)
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

# # 测试数据
# X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
#                   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
# K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
# K_multi = torch.stack((K, K + 1, K + 2), 0)
#
# print(f"输入 X 形状: {X.shape}")      # torch.Size([2, 3, 3])
# print(f"卷积核 K 形状: {K_multi.shape}") # torch.Size([3, 2, 2, 2])
# print(f"输出结果形状: {corr2d_multi_in_out(X, K_multi).shape}") # torch.Size([3, 2, 2])

def corr2d_multi_in_out_1x1 (X, K):
    c_i ,h,w=X.shape #均为3
    c_o=K.shape[0] #2
    X=X.reshape((c_i,h*w)) #3*9
    K=K.reshape((c_o,c_i)) # 2*3 卷积核的维度为c_o*c_i*kh*kw 那么因此K是一个1*1的卷积核
    Y=torch.matmul(K,X) #2*9的矩阵
    return Y.reshape((c_o,h,w)) #2*3*3的矩阵

X = torch.normal(0, 1, (3, 3, 3))

K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print(float(torch.abs(Y1 - Y2).sum()) < 1e-6)

import os
# 必须在导入任何可能使用 OpenMP 的库之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt




# ##观察不同方向梯度差异很大的函数的梯度下降
# ### s1 和 s2 通常是用来表示梯度的分量（即函数对 x1、x2 的偏导数）
# # eta=4
# def f_2d(x1,x2):
#     return 0.1*x1**2+2*x2**2
#
# def gd_2d(x1,x2,s1,s2):
#     return (x1-eta*0.2*x1,x2-eta*4*x2,0,0)
#
#
# def momentum_2d(x1,x2,v1,v2):
#     v1=beta*v1+0.2*x1
#     v2=beta*v2+4*x2
#     return x1-eta*v1,x2-eta*v2,v1,v2
#
# eta=0.6
# beta=0.5
#
#
# d2l.show_trace_2d(f_2d,d2l.train_2d(momentum_2d))
# plt.show()



### 动量法需要维护速度，即states
### 初始化状态参数
def init_momentum_states(feature_dim):
    v_w=torch.zeros((feature_dim,1))
    v_b=torch.zeros(1)
    return (v_w,v_b)


def sgd_momentum(params,states,hyperparams):
    for p,v in zip(params,states):
        with torch.no_grad():
            v[:]=hyperparams['momentum']*v+p.grad  ## p代表自变量
            p[:]-=hyperparams['lr']*v
        p.grad.data.zero_()



def train_mometum(lr,momentum,num_epochs=2):
    d2l.train_ch11(sgd_momentum,init_momentum_states(feature_dim),
                   {'lr':lr,'momentum':momentum},data_iter,
                   feature_dim,num_epochs)


data_iter,feature_dim=d2l.get_data_ch11(batch_size=10)
train_mometum(0.02,0.5)
plt.show()
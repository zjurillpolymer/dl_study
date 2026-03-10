import os
# 必须在导入任何可能使用 OpenMP 的库之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



import math
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

def rmsprop_2d(x1,x2,s1,s2):
    g1,g2,eps=0.2*x1,4*x2,1e-6
    s1=gamma*s1+(1-gamma)*g1**2
    s2= gamma * s2 + (1 - gamma) * g2 ** 2
    x1-=eta/math.sqrt(s1+eps)*g1
    x2-=-eta/math.sqrt(s2+eps)*g2
    return x1,x2,s1,s2


def f_2d(x1,x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


eta,gamma=0.4,1
d2l.show_trace_2d(f_2d,d2l.train_2d(rmsprop_2d))
plt.show()
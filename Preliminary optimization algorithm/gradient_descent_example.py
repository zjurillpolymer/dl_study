import os
# 必须在导入任何可能使用 OpenMP 的库之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt
# def f(x):
#     return x**2
#
# def f_grad(x):
#     return 2*x
c=torch.tensor(0.15*np.pi)


def train_2d(trainer,steps=20,f_grad=None):
    x1,x2,s1,s2=-5,-2,0,0
    results=[(x1,x2)]
    for i in range(steps):
        if f_grad:
            x1,x2,s1,s2=trainer(x1,x2,s1,s2,f_grad)

        else:
            x1,x2,s1,s2=trainer(x1,x2,s1,s2)
        results.append((x1,x2))
    return results




def f(x):
    return x*torch.cos(c*x)


def f_grad(x):
    return torch.cos(c*x)-c*x*torch.sin(c*x)

def gd(eta,f_grad):
    x=10.0
    results=[x]
    for i in range(10):
        x=x-eta*f_grad(x)
        results.append(x)
    return results
# results=gd(1,f_grad)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / abs(f_hess(x))
        results.append(float(x))
    print('epoch 10, x:', x)
    return results


def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.01)
    d2l.set_figsize((10.0,10.0))
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])
    plt.show()

# show_trace(results, f)

def show_trace_2d(f, results):  #@save
    """显示优化过程中2D变量的轨迹"""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
                          torch.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
    plt.show()


c = torch.tensor(0.15 * np.pi)

def f(x):  # 目标函数
    return x * torch.cos(c * x)

def f_grad(x):  # 目标函数的梯度
    return torch.cos(c * x) - c * x * torch.sin(c * x)

def f_hess(x):  # 目标函数的Hessian
    return - 2 * c * torch.sin(c * x) - x * c**2 * torch.cos(c * x)

show_trace(newton(), f)




show_trace(newton(0.5), f)





import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

train_end = 800
x_train = x[:train_end]
x_test = x[train_end:]

p_values = [5, 10, 20, 50, 100]
errors = []

for p in p_values:
    # 构造训练数据
    X, y = [], []
    for t in range(p, len(x_train)):
        X.append(x_train[t - p:t])
        y.append(x_train[t])
    X = torch.stack(X)
    y = torch.stack(y)

    # 线性回归
    A = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
    w = torch.linalg.lstsq(A, y).solution
    ar_weights = w[:-1]
    bias = w[-1]

    # 单步预测（测试集）
    preds = []
    history = x_train[-p:].clone()
    for i in range(len(x_test)):
        pred = history[-p:] @ ar_weights + bias
        preds.append(pred.item())
        history = torch.cat([history, x_test[i:i + 1]])  # 用真实值更新

    preds = torch.tensor(preds)
    mse = torch.mean((preds - x_test) ** 2).item()
    errors.append(mse)

# 绘图
plt.plot(p_values, errors, 'o-')
plt.xlabel('AR order (p)')
plt.ylabel('Test MSE (1-step)')
plt.title('Prediction Error vs. History Length')
plt.grid(True)
plt.show()
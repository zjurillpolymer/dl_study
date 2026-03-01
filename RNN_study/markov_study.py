import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import matplotlib.pyplot as plt

# ----------------------------
# 1. 生成数据
# ----------------------------
T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

# 划分训练/测试（比如最后 100 点用于测试）
train_end = 800
test_start = train_end
x_train = x[:train_end]
x_test = x[test_start:]

# ----------------------------
# 2. 训练 AR(p) 模型
# ----------------------------
p = 20  # 使用过去20个点
X, y = [], []
for t in range(p, len(x_train)):
    X.append(x_train[t - p:t])
    y.append(x_train[t])
X = torch.stack(X)  # shape: (N, p)
y = torch.stack(y)  # shape: (N,)

# 线性回归（带偏置）
A = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)  # (N, p+1) 增广算偏置
w = torch.linalg.lstsq(A, y).solution
ar_weights = w[:-1]  # shape: (p,)
bias = w[-1]

print(f"AR({p}) 模型训练完成。")

# ----------------------------
# 3. 单步预测（on test set）
# ----------------------------
single_step_preds = []
history = x_train[-p:].clone()  # 最近 p 个真实值作为初始历史

for i in range(len(x_test)):
    # 用真实历史预测下一步
    pred = history[-p:] @ ar_weights + bias
    single_step_preds.append(pred.item())
    # 更新历史：加入真实值（不是预测值！）
    history = torch.cat([history, x_test[i:i+1]])

single_step_preds = torch.tensor(single_step_preds)

# ----------------------------
# 4. k步滚动预测（例如预测未来50步）
# ----------------------------
k = 50
multi_step_preds = []
history_multi = x_train[-p:].clone()  # 从训练集末尾开始

for i in range(k):
    pred = history_multi[-p:] @ ar_weights + bias
    multi_step_preds.append(pred.item())
    # 关键：用预测值更新历史（不是真实值！）
    history_multi = torch.cat([history_multi, pred.unsqueeze(0)])

multi_step_preds = torch.tensor(multi_step_preds)
time_multi = time[test_start:test_start + k]

# ----------------------------
# 5. 可视化
# ----------------------------
plt.figure(figsize=(14, 6))

# 单步预测
plt.subplot(1, 2, 1)
plt.plot(time[test_start:test_start + len(x_test)], x_test, label='Ground Truth', alpha=0.7)
plt.plot(time[test_start:test_start + len(x_test)], single_step_preds, '--', label='1-Step Prediction')
plt.title('1-Step-Ahead Prediction')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.legend()

# 多步预测
plt.subplot(1, 2, 2)
plt.plot(time[test_start:test_start + k], x_test[:k], label='Ground Truth', alpha=0.7)
plt.plot(time_multi, multi_step_preds, '--', label=f'{k}-Step Recursive Prediction')
plt.title(f'{k}-Step Recursive Prediction')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.legend()
plt.show()
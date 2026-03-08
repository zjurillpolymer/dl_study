import torch
import numpy as np

# ====================== 1. 生成模拟数据并转为Tensor ======================
# 固定随机种子，结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 真实参数：w=2, b=1，加噪声
X_np = np.random.rand(1000, 1)  # 1000个样本，特征维度1
y_np = 2 * X_np + 1 + 0.1 * np.random.randn(1000, 1)

# 转为PyTorch张量，float32是PyTorch默认的计算精度
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)


# ====================== 2. PyTorch版MBGD实现 ======================
def mini_batch_gradient_descent_torch(X, y, batch_size=32, lr=0.1, epochs=50):
    """
    PyTorch实现小批量梯度下降（自动计算梯度）
    参数：
        X: 特征张量 (n_samples, n_features)
        y: 标签张量 (n_samples, 1)
        batch_size: 批次大小
        lr: 学习率
        epochs: 迭代轮数
    返回：
        w, b: 训练后的权重和偏置
    """
    n_samples, n_features = X.shape

    # 1. 定义可训练参数（设置requires_grad=True，标记需要求导）
    w = torch.zeros((n_features, 1), dtype=torch.float32, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    for epoch in range(epochs):
        # 2. 打乱数据（PyTorch方式）
        indices = torch.randperm(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # 3. 遍历所有小批量
        for i in range(0, n_samples, batch_size):
            # 取当前批次数据
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # 4. 前向计算：y_hat = wX + b
            y_hat = torch.matmul(X_batch, w) + b

            # 5. 计算损失（MSE，和NumPy版本一致）
            loss = torch.mean((y_hat - y_batch) ** 2) / 2

            # 6. 反向传播：自动计算梯度（核心！无需手动推导dw/db）
            loss.backward()

            # 7. 更新参数：必须关闭梯度计算（避免更新过程被记录）
            with torch.no_grad():
                w -= lr * w.grad  # w.grad就是自动计算的权重梯度
                b -= lr * b.grad  # b.grad就是自动计算的偏置梯度

                # 8. 清空梯度（关键！否则下一轮会累积梯度）
                w.grad.zero_()
                b.grad.zero_()

        # 每10轮打印进度
        if (epoch + 1) % 10 == 0:
            # 计算全量数据的损失（评估收敛情况）
            y_pred = torch.matmul(X, w) + b
            total_loss = torch.mean((y_pred - y) ** 2) / 2
            print(f"Epoch {epoch + 1}, Loss: {total_loss.item():.6f}, w: {w.item():.4f}, b: {b.item():.4f}")

    # 返回参数的数值（脱离计算图）
    return w.item(), b.item()


# ====================== 3. 运行训练 ======================
w_trained, b_trained = mini_batch_gradient_descent_torch(X, y, batch_size=32, lr=0.1, epochs=50)

# 输出最终结果
print("\n训练完成：")
print(f"真实w=2，训练后w={w_trained:.4f}")
print(f"真实b=1，训练后b={b_trained:.4f}")

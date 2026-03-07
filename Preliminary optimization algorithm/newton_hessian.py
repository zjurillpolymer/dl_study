import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 定义函数、梯度、Hessian =====================
def f(x):
    """目标函数: f(x1,x2) = 1000x1² + x2²"""
    x1, x2 = x
    return 1000 * x1**2 + x2**2

def gradient(x):
    """梯度: [2000x1, 2x2]"""
    x1, x2 = x
    return np.array([2000 * x1, 2 * x2])

def hessian(x):
    """Hessian矩阵: [[2000, 0], [0, 2]]"""
    return np.array([[2000, 0], [0, 2]])

# ===================== 2. 定义迭代算法 =====================
def gradient_descent(x_init, lr=0.0005, max_iter=10000, tol=1e-6):
    """普通梯度下降（轻量牛顿法，无预处理）"""
    x = np.array(x_init, dtype=np.float64)
    loss_history = [f(x)]
    for _ in range(max_iter):
        grad = gradient(x)
        # 收敛判断：函数值小于tol则停止
        if f(x) < tol:
            break
        x = x - lr * grad
        loss_history.append(f(x))
    return np.array(loss_history)

def preconditioned_newton(x_init, lr=1.0, max_iter=10000, tol=1e-6):
    """预处理牛顿法（对角Hessian绝对值）"""
    x = np.array(x_init, dtype=np.float64)
    loss_history = [f(x)]
    eps = 1e-8  # 防止除0
    for _ in range(max_iter):
        grad = gradient(x)
        hess = hessian(x)
        # 提取对角Hessian并取绝对值
        diag_hess = np.diag(hess)
        precond = np.abs(diag_hess) + eps
        # 缩放梯度
        scaled_grad = grad / precond
        # 收敛判断
        if f(x) < tol:
            break
        x = x - lr * scaled_grad
        loss_history.append(f(x))
    return np.array(loss_history)

# ===================== 3. 实验设置 =====================
x_init = [1.0, 1.0]  # 初始点
tol = 1e-6           # 收敛阈值（函数值<1e-6则认为收敛）
max_iter = 10000     # 最大迭代次数

# 普通梯度下降的学习率：需选折中值（0.0005，避免x1震荡）
lr_gd = 0.0005
# 预处理牛顿法的学习率：可设为1.0（缩放后梯度已适配维度）
lr_pre = 1.0

# 运行算法
loss_gd = gradient_descent(x_init, lr=lr_gd, max_iter=max_iter, tol=tol)
loss_pre = preconditioned_newton(x_init, lr=lr_pre, max_iter=max_iter, tol=tol)

# ===================== 4. 可视化对比 =====================
plt.rcParams['font.size'] = 12
plt.figure(figsize=(10, 6))

# 绘制损失曲线（对数坐标，更清晰展示收敛速度）
plt.plot(range(len(loss_gd)), np.log10(loss_gd), label='without pre', linewidth=2)
plt.plot(range(len(loss_pre)), np.log10(loss_pre), label='newton-Hessian', linewidth=2)

# 标注关键信息
plt.xlabel('epochs')
plt.ylabel('log10(f)')
plt.title('rate')
plt.legend()
plt.grid(True, alpha=0.3)

# 打印收敛迭代次数
print(f"普通梯度下降收敛迭代次数：{len(loss_gd)}")
print(f"预处理牛顿法收敛迭代次数：{len(loss_pre)}")
print(f"预处理牛顿法的收敛速度提升倍数：{len(loss_gd)/len(loss_pre):.2f}倍")

plt.show()

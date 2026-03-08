import os
# 必须在导入任何可能使用 OpenMP 的库之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import torch
import matplotlib.pyplot as plt


# ================= 1. 定义目标函数和梯度 =================

def f(x1, x2):
    """目标函数: f(x1, x2) = x1^2 + 2*x2^2"""
    return x1 ** 2 + 2 * x2 ** 2


def f_grad(x1, x2):
    """真实梯度"""
    return 2 * x1, 4 * x2


# ================= 2. 定义优化器 (带噪声的 SGD) =================

def sgd(x1, x2, s1, s2, f_grad):
    """
    带高斯噪声的随机梯度下降
    返回: (new_x1, new_x2, new_s1, new_s2)
    """
    g1, g2 = f_grad(x1, x2)

    # 添加噪声: N(0, 1)
    # 使用 .item() 转换为标量以便与 float 运算，或者保持 tensor 均可，这里统一为 float
    # noise1 = torch.normal(0.0, 1.0, size=(1,)).item()
    # noise2 = torch.normal(0.0, 1.0, size=(1,)).item()
    #
    # g1 += noise1
    # g2 += noise2

    eta_t = eta * lr()

    x1_new = x1 - eta_t * g1
    x2_new = x2 - eta_t * g2

    # s1, s2 在普通 SGD 中不使用，直接返回 0
    return x1_new, x2_new, 0.0, 0.0


def constant_lr():
    return 1.0


# 全局超参数
eta = 0.1  # 基础学习率
lr = constant_lr


# ================= 3. 修复后的 train_2d =================

def train_2d(optimizer_fn, steps=50, initial_state=(5.0, 10.0)):
    """
    通用 2D 训练函数
    :param optimizer_fn: 优化器函数
    :param steps: 迭代步数
    :param initial_state: 初始状态，可以是 (x1, x2) 或 (x1, x2, s1, s2)
    :return: 包含每一步 (x1, x2) 元组的列表
    """
    # 修复：根据传入参数的长度自动补全 s1, s2
    if len(initial_state) == 2:
        x1, x2 = initial_state
        s1, s2 = 0.0, 0.0
    elif len(initial_state) == 4:
        x1, x2, s1, s2 = initial_state
    else:
        raise ValueError("initial_state must be a tuple of length 2 or 4")

    # 记录轨迹，包含初始点
    trajectory = [(x1, x2)]

    for i in range(steps):
        # 调用优化器更新状态
        x1, x2, s1, s2 = optimizer_fn(x1, x2, s1, s2, f_grad)
        trajectory.append((x1, x2))

    return trajectory


# ================= 4. show_trace_2d =================

def show_trace_2d(trajectory, result_func=f):
    """
    美化版 2D 优化轨迹可视化
    """
    # 1. 准备网格数据
    x1_range = torch.linspace(-6, 6, 200)  # 更高分辨率
    x2_range = torch.linspace(-6, 6, 200)
    X1, X2 = torch.meshgrid(x1_range, x2_range)
    Y = result_func(X1, X2)

    # 2. 创建图形
    fig, ax = plt.subplots(figsize=(9, 7))

    # 绘制填充等高线（更柔和的背景）
    contourf = ax.contourf(X1.numpy(), X2.numpy(), Y.numpy(), levels=40, cmap='Blues', alpha=0.3)

    # 绘制轮廓线（少量关键等高线）
    contour = ax.contour(X1.numpy(), X2.numpy(), Y.numpy(), levels=[1, 4, 9, 16, 25, 36, 49, 64],
                         colors='gray', linewidths=1.2, linestyles='--', alpha=0.7)

    # 可选：添加等高线标签（仅在大值处）
    # ax.clabel(contour, inline=True, fontsize=8, fmt='%1.0f', manual=[(0,0), (2,2), (-2,-2)])

    # 3. 提取轨迹坐标（降采样：每5步取一个点，避免 overcrowding）
    traj_x1 = [t[0] for t in trajectory[::5]]  # 每隔5步取一个点
    traj_x2 = [t[1] for t in trajectory[::5]]

    # 原始完整轨迹用于连线（半透明细线）
    full_x1 = [t[0] for t in trajectory]
    full_x2 = [t[1] for t in trajectory]

    # 绘制完整轨迹（淡红色细线）
    ax.plot(full_x1, full_x2, 'r-', linewidth=1.5, alpha=0.4, label='Full Path')

    # 绘制采样点（醒目圆点）
    ax.plot(traj_x1, traj_x2, 'o', color='#e74c3c', markersize=8, markeredgecolor='white',
            markeredgewidth=1.5, label='Sampled Points (every 5 steps)')

    # 标记起点和终点（更大、更醒目）
    ax.plot(trajectory[0][0], trajectory[0][1], 'X', color='#3498db', markersize=14,
            markeredgewidth=2.5, label='Start')
    ax.plot(trajectory[-1][0], trajectory[-1][1], '*', color='#2ecc71', markersize=18,
            markeredgewidth=2.5, label='End')

    # 4. 设置图表属性
    ax.set_xlabel('$x_1$', fontsize=12, fontweight='bold')
    ax.set_ylabel('$x_2$', fontsize=12, fontweight='bold')
    ax.set_title(f'SGD with Noise — Trajectory over {len(trajectory) - 1} Steps (η={eta})',
                 fontsize=14, pad=15, fontweight='bold')

    # 图例美化
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                       facecolor='white', edgecolor='gray', fontsize=10)
    legend.get_frame().set_alpha(0.9)

    # 网格 & 背景
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.4, color='#cccccc')
    ax.set_axisbelow(True)  # 网格在底层

    # 坐标轴比例 & 范围
    ax.set_aspect('equal')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    # 去掉顶部和右边框（现代风格）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    # 添加轻微阴影效果（可选）
    # from matplotlib.patches import Rectangle
    # rect = Rectangle((-6, -6), 12, 12, fill=True, facecolor='none',
    #                  edgecolor='black', linewidth=0.5, alpha=0.1)
    # ax.add_patch(rect)

    plt.tight_layout()
    plt.show()


# ================= 5. 执行可视化 =================

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)

    print("正在运行带噪声的 SGD 训练...")
    # 现在可以安全地传入 2 个值的元组
    trajectory = train_2d(sgd, steps=50, initial_state=(5.0, 10.0))

    print("正在绘制轨迹...")
    show_trace_2d(trajectory)
import os
# 必须在导入任何可能使用 OpenMP 的库之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from mpl_toolkits import mplot3d
from d2l import torch as d2l


def f(x):
    return x*torch.cos(np.pi*x)


def g(x):
    return f(x)+0.2*torch.cos(5*np.pi*x)


def annotate(text, xy, xytext): #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                            arrowprops=dict(arrowstyle='->'))
    d2l.plt.show()
# x = torch.arange(0.5, 1.5, 0.01)
# d2l.set_figsize((4.5, 2.5))
# d2l.plot(x, [f(x), g(x)], 'x', 'risk')
# annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
# annotate('min of risk', (1.1, -1.05), (0.95, -0.5))


# x = torch.arange(-1.0, 2.0, 0.01)
# d2l.plot(x, [f(x), ], 'x', 'f(x)')
# annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
# annotate('global minimum', (1.1, -0.95), (0.6, 0.8))



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具包

# ---------------------- 1. 生成数据 ----------------------
# 生成x和y的取值范围，从-5到5，共100个均匀分布的点
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

# 将x和y转换为网格矩阵（二维数组），用于计算每个坐标点的z值
X, Y = np.meshgrid(x, y)

# 计算z = x² - y²
Z = X**2 - Y**2

# ---------------------- 2. 创建绘图对象 ----------------------
# 创建画布
fig = plt.figure(figsize=(8, 6))
# 添加3D坐标轴
ax = fig.add_subplot(111, projection='3d')

# ---------------------- 3. 绘制3D曲面图 ----------------------
# 绘制曲面，cmap指定颜色映射（彩虹色），alpha设置透明度
surf = ax.plot_surface(X, Y, Z, cmap='rainbow', alpha=0.8)

# ---------------------- 4. 美化图表 ----------------------
# 添加颜色条（右侧），用于解释颜色对应的z值大小
fig.colorbar(surf, shrink=0.5, aspect=5)

# 设置坐标轴标签
ax.set_xlabel('X Axis', fontsize=10)
ax.set_ylabel('Y Axis', fontsize=10)
ax.set_zlabel('Z Axis (x² - y²)', fontsize=10)

# 设置图表标题
ax.set_title('3D Plot of z = x² - y² (Saddle Surface)', fontsize=12)

# ---------------------- 5. 显示图表 ----------------------
plt.show()

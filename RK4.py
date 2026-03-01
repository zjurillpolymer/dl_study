import numpy as np
import matplotlib.pyplot as plt
def rk4(f, y0, t0, t_end, h):
    t_values = np.arange(t0, t_end + h, h)  # 包含 t_end
    n = len(t_values)

    if np.isscalar(y0):
        y_values = np.zeros(n)
        y_values[0] = y0
    else:
        y_values = np.zeros((n, len(y0))) ##
        y_values[0] = y0  # ← 修正：索引应为 0

    for i in range(n - 1):
        t = t_values[i]
        y = y_values[i]
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h,   y + h   * k3)

        # ← 修正：去掉多余的 h
        y_values[i+1] = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6

    return t_values, y_values


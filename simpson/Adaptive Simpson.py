def adaptive_simpson(f, a, b, tol=1e-6, max_depth=20):
    """
    自适应 Simpson 方法

    参数:
        f: 被积函数
        a, b: 积分区间
        tol: 容差
        max_depth: 最大递归深度

    返回:
        积分近似值
    """

    def recursive_simpson(a, b, fa, fb, fc, depth):
        # 计算左右两半的 Simpson 值
        c = (a + b) / 2
        d = (a + c) / 2
        e = (c + b) / 2

        fd = f(d)
        fe = f(e)

        # 整个区间的 Simpson 值
        S1 = (b - a) / 6 * (fa + 4 * fc + fb)
        # 两个子区间的 Simpson 值之和
        S2 = (b - a) / 12 * (fa + 4 * fd + 2 * fc + 4 * fe + fb)

        # 误差估计
        error = abs(S2 - S1) / 15

        if error < tol or depth >= max_depth:
            # Richardson 外推提高精度
            return S2 + (S2 - S1) / 15
        else:
            # 递归计算
            left = recursive_simpson(a, c, fa, fc, fd, depth + 1)
            right = recursive_simpson(c, b, fc, fb, fe, depth + 1)
            return left + right

    # 初始计算
    fa = f(a)
    fb = f(b)
    fc = f((a + b) / 2)

    return recursive_simpson(a, b, fa, fb, fc, 0)


# 示例
import math


def f(x):
    return math.sin(x) / x if x != 0 else 1


result = adaptive_simpson(f, 0, 10, tol=1e-10)
print(f"自适应 Simpson: ∫sin(x)/x dx from 0 to 10 ≈ {result:.10f}")
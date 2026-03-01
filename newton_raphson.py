import math

def newton_raphson(f, f_prime, x0, x_tol=1e-10, f_tol=1e-10, maxiter=100):
    if not callable(f) or not callable(f_prime):
        raise TypeError("f 和 f_prime 必须是可调用对象")

    x = float(x0)  # 确保是浮点数
    history = [x]

    for i in range(maxiter):
        try:
            fx = f(x)
            fpx = f_prime(x)
        except Exception as e:
            print(f"函数求值出错: {e}")
            return None, history

        if not math.isfinite(fx) or not math.isfinite(fpx):
            print(f"警告: 在 x = {x} 处函数值或导数非有限")
            return None, history

        if abs(fpx) < 1e-15:
            print(f"警告: 在 x = {x} 处导数接近零")
            return None, history

        if abs(fpx) < 1e-15 * max(1.0, abs(x)):  # 相对+绝对混合判断
            print(f"警告: 导数过小，可能遇到重根或奇点")
            return None, history

        x_new = x - fx / fpx
        history.append(x_new)

        # 防止发散
        if abs(x_new) > 1e15:
            print(f"警告: 迭代发散，x = {x_new}")
            return None, history

        # 收敛判断：同时看 x 变化和 f(x) 大小
        if abs(x_new - x) < x_tol and abs(f(x_new)) < f_tol:
            return x_new, history

        x = x_new

    print(f"警告: 达到最大迭代次数 ({maxiter}) 仍未收敛")
    return None, history
def composite_simpson_13(f, a,b,n):
    if(n%2!=0):
        raise ValueError("n 必须是偶数")

    h=(b-a)/n
    x=[a+i*h for i in range(n+1)]

    sum_odd = 0
    sum_even = 0

    # 奇数下标项 (1, 3, 5, ..., n-1)
    for i in range(1, n, 2):
        sum_odd += f(x[i])

    # 偶数下标项 (2, 4, 6, ..., n-2)
    for i in range(2, n - 1, 2):
        sum_even += f(x[i])

    # Simpson 公式
    integral = (h / 3) * (f(a) + f(b) + 4 * sum_odd + 2 * sum_even)
    return integral



def test_function(x):
    import math
    return math.exp(-x**2)

# 计算 ∫e^(-x²)dx from 0 to 1
for n in [4, 8, 16, 32]:
    result = composite_simpson_13(test_function, 0, 1, n)
    print(f"n={n}: {result:.8f}")
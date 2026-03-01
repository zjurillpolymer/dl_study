def simpson_13_single(f,a,b):
    midpoint = (a + b) / 2
    return (b - a) / 6 * (f(a) + 4 * f(midpoint) + f(b))

def f(x):
    return x**2

result = simpson_13_single(f, 0, 2)
print(f"∫x²dx from 0 to 2 ≈ {result}")  ##三取样的Simpson积分
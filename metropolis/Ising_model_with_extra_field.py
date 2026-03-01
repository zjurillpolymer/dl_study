import numpy as np
import matplotlib.pyplot as plt


def calc_energy(lattice, h=0.0):
    N = lattice.shape[0]
    interaction = 0
    for i in range(N):
        for j in range(N):
            s = lattice[i, j]
            neighbors = lattice[(i + 1) % N, j] + lattice[i, (j + 1) % N]
            interaction += -s * neighbors
    interaction /= (N * N)
    field_energy = -h * np.mean(lattice)
    return interaction + field_energy

def calc_mag(lattice):
    """计算磁化强度"""
    return np.mean(lattice)


def simulate_ising(N, T, h=0.0, steps=2000, burn_in=1000):
    """带外场的Ising模型模拟"""
    lattice = np.random.choice([1, -1], size=(N, N))
    energies = []
    magnets = []

    for s in range(steps):
        # 一个蒙特卡洛步：尝试翻转 N×N 个自旋
        for _ in range(N * N):
            i, j = np.random.randint(0, N), np.random.randint(0, N)

            # 计算4个最近邻自旋的和
            sn = (lattice[(i + 1) % N, j] + lattice[(i - 1) % N, j] +
                  lattice[i, (j + 1) % N] + lattice[i, (j - 1) % N])

            # 能量变化：包含外场项
            dE = 2 * lattice[i, j] * (sn + h)

            # Metropolis准则
            if dE <= 0 or np.random.rand() < np.exp(-dE / T):
                lattice[i, j] *= -1

        # 热平衡后收集数据
        if s >= burn_in:
            energies.append(calc_energy(lattice, h))
            magnets.append(calc_mag(lattice))

    return np.array(energies), np.array(magnets), lattice



# 测试不同外场下的行为
temps = np.linspace(1.5, 3.0, 20)
hs = [0.0, 0.1, 0.2, 0.5]  # 不同外场强度

results = {}
for h in hs:
    avg_mags = []
    for T in temps:
        energies, magnets, _ = simulate_ising(20, T, h=h, steps=1500, burn_in=500)
        avg_mags.append(np.mean(np.abs(magnets)))  # 平均磁化强度绝对值
    results[h] = avg_mags

# 绘制磁化曲线
plt.figure(figsize=(10, 6))
for h, mags in results.items():
    plt.plot(temps, mags, 'o-', label=f'h = {h}')
plt.xlabel('Temperature T')
plt.ylabel(r'$\langle |M| \rangle$')
plt.title('Ising Model with External Field')
plt.legend()
plt.grid(True)
plt.show()
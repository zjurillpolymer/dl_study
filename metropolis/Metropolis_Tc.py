import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def simulate_ising(N, T, steps, burn_in=500):
    lattice = np.random.choice([-1, 1], size=(N, N))

    # 计算初始能量（正确计算：每个键只算一次）
    E_total = 0.0
    for i in range(N):
        for j in range(N):
            s = lattice[i, j]
            nb = (lattice[(i + 1) % N, j] + lattice[(i - 1) % N, j] +
                  lattice[i, (j + 1) % N] + lattice[i, (j - 1) % N])
            E_total += -s * nb
    E_total /= 2  # 每个键被计算了两次
    M_total = np.sum(lattice)

    energies = []
    mags = []

    for step in range(steps):
        for _ in range(N * N):
            i, j = np.random.randint(0, N), np.random.randint(0, N)
            sn = (lattice[(i + 1) % N, j] + lattice[(i - 1) % N, j] +
                  lattice[i, (j + 1) % N] + lattice[i, (j - 1) % N])
            dE = 2 * lattice[i, j] * sn

            if dE <= 0 or np.random.rand() < np.exp(-dE / T):
                lattice[i, j] *= -1
                E_total += dE
                M_total += 2 * lattice[i, j]  # 注意：翻转后磁矩变化是2倍新值

        # 只在达到热平衡后收集数据
        if step >= burn_in:
            energies.append(E_total / (N * N))
            mags.append(M_total / (N * N))

    # 将列表转换为数组（在循环外部！）
    energies = np.array(energies)
    mags = np.array(mags)

    # 计算统计量
    avg_E = np.mean(energies)
    avg_E2 = np.mean(energies ** 2)
    avg_M_abs = np.mean(np.abs(mags))
    avg_M2 = np.mean(mags ** 2)

    # 正确计算磁化率（使用mags数组）
    chi = (N * N) * (avg_M2 - avg_M_abs ** 2) / T
    C = (N * N) * (avg_E2 - avg_E ** 2) / (T ** 2)

    return {
        'T': T,
        '<|M|>': avg_M_abs,
        'chi': chi,
        'C': C,
        'mags': mags,
        'energies': energies
    }


if __name__ == '__main__':
    N = 10
    steps = 2000  # 增加步数以获得更好的统计
    burn_in = 1000
    temps = np.linspace(2.0, 2.6, 25)

    results = []
    for T in temps:
        res = simulate_ising(N, T, steps, burn_in)
        results.append(res)
        print(f"T = {T:.3f} → <|M|> = {res['<|M|>']:.4f}, χ = {res['chi']:.2f}")

    Ts = np.array([r['T'] for r in results])
    mag_abs = np.array([r['<|M|>'] for r in results])
    chis = np.array([r['chi'] for r in results])
    Cs = np.array([r['C'] for r in results])

    # 高斯拟合
    # p0是初始猜测参数
    idx_max = np.argmax(chis)
    p0 = [chis[idx_max], Ts[idx_max], 0.1]

    try:
        popt_chi, _ = curve_fit(gaussian, Ts, chis, p0=p0)
        Tc_chi = popt_chi[1]
    except:
        Tc_chi = Ts[idx_max]
        print("Gaussian fit failed for χ, using max position.")
        popt_chi = [chis[idx_max], Ts[idx_max], 0.1]  # 添加默认值用于绘图

    idx_max_C = np.argmax(Cs)
    p0_C = [Cs[idx_max_C], Ts[idx_max_C], 0.1]
    try:
        popt_C, _ = curve_fit(gaussian, Ts, Cs, p0=p0_C)
        Tc_C = popt_C[1]
    except:
        Tc_C = Ts[idx_max_C]
        print("Gaussian fit failed for C, using max position.")
        popt_C = [Cs[idx_max_C], Ts[idx_max_C], 0.1]  # 添加默认值用于绘图

    # 最终估计：取平均
    Tc_est = (Tc_chi + Tc_C) / 2

    # --- 绘图 ---
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(Ts, mag_abs, 'bo-')
    plt.axvline(Tc_est, color='r', linestyle='--', label=f'Estimated $T_c$ = {Tc_est:.3f}')
    plt.title(r"$\langle |M| \rangle$ vs $T$")
    plt.xlabel("Temperature $T$")
    plt.ylabel(r"$\langle |M| \rangle$")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(Ts, chis, 'go-', label='Data')
    Ts_fine = np.linspace(Ts.min(), Ts.max(), 200)
    plt.plot(Ts_fine, gaussian(Ts_fine, *popt_chi), 'r--', label=f'Fit: $T_c$ = {Tc_chi:.3f}')
    plt.title(r"Magnetic Susceptibility $\chi$ vs $T$")
    plt.xlabel("Temperature $T$")
    plt.ylabel(r"$\chi$")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(Ts, Cs, 'mo-', label='Data')
    plt.plot(Ts_fine, gaussian(Ts_fine, *popt_C), 'r--', label=f'Fit: $T_c$ = {Tc_C:.3f}')
    plt.title(r"Specific Heat $C$ vs $T$")
    plt.xlabel("Temperature $T$")
    plt.ylabel(r"$C$")
    plt.legend()

    plt.tight_layout()
    plt.savefig("ising_tc_estimation.png", dpi=150)
    plt.show()

    # --- 输出结果 ---
    print("\n" + "=" * 50)
    print(f"Exact Tc (Onsager):       {2 / np.log(1 + np.sqrt(2)):.5f}")
    print(f"Estimated Tc (χ peak):    {Tc_chi:.5f}")
    print(f"Estimated Tc (C peak):    {Tc_C:.5f}")
    print(f"Final estimated Tc:       {Tc_est:.5f}")
    print(f"Error:                    {abs(Tc_est - 2 / np.log(1 + np.sqrt(2))):.5f}")
    print("=" * 50)
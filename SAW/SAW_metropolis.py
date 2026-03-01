import numpy as np
import matplotlib.pyplot as plt


def initialize_chain(L, N):
    """在 N×N 格子上生成一条长度为 L 的自避行走"""
    chain = np.zeros((L, 2), dtype=int)
    chain[0] = [N // 2, N // 2]  # 从中心开始

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for i in range(1, L):
        valid = False
        attempts = 0
        while not valid and attempts < 100:
            dx, dy = directions[np.random.randint(4)]
            new_pos = (chain[i - 1] + [dx, dy]) % N  # 周期性边界
            # 检查是否与已有单体重叠
            if not any(np.array_equal(new_pos, chain[j]) for j in range(i)):
                chain[i] = new_pos
                valid = True
            attempts += 1
        if not valid:
            raise RuntimeError("Failed to initialize chain!")
    return chain


def metropolis_step(chain, N, T=1.0):
    L = len(chain)
    if L < 3:
        return chain

    i = np.random.randint(1, L-1)          # 选中间单体
    prev = chain[i-1]
    curr = chain[i]
    nextp = chain[i+1]                      # next 是关键字，改名叫 nextp

    # 计算从 prev → curr 的向量
    vec_in = (curr - prev) % N

    # 两个可能的 90° 旋转方向
    rot_cw  = np.array([ vec_in[1], -vec_in[0]]) % N
    rot_ccw = np.array([-vec_in[1],  vec_in[0]]) % N

    possible_new = []
    # 尝试两个方向
    for rot in [rot_cw, rot_ccw]:
        candidate = (prev + rot) % N
        # 不能回到原位置（避免无效移动）
        if not np.all(candidate == curr):
            possible_new.append(candidate)

    if not possible_new:
        return chain

    # 随机选一个尝试
    new_pos = possible_new[np.random.randint(len(possible_new))]

    # 检查新位置是否被占用（只排除自己当前位置）
    occupied = {tuple(p) for p in chain}
    occupied.remove(tuple(curr))

    if tuple(new_pos) in occupied:
        return chain  # 撞到了

    # 纯 SAW 无能量差 → 永远接受（只要不重叠）
    chain[i] = new_pos
    return chain


def calc_rg(chain, N):
    # 处理周期性边界：将链“展开”到最小图像
    # 简化：假设链不 wrap around（小链可接受）
    r_cm = np.mean(chain, axis=0)
    rg2 = np.mean(np.sum((chain - r_cm)**2, axis=1))
    return np.sqrt(rg2)


def simulate_polymer(L, N, steps, burn_in=500):
    chain = initialize_chain(L, N)
    rg_list = []

    for step in range(steps):
        chain = metropolis_step(chain, N)
        if step >= burn_in:
            rg_list.append(calc_rg(chain, N))

    return np.array(rg_list), chain


# 运行
L = 20  # 聚合物长度
N = 400  # 格子大小（要远大于链尺寸）
steps = 1000

rgs, final_chain = simulate_polymer(L, N, steps)

print(f"Average Rg = {np.mean(rgs):.2f}")

# 可视化最终构型
plt.figure(figsize=(6, 6))
plt.plot(final_chain[:, 0], final_chain[:, 1], 'o-', markersize=5)
plt.xlim(0, N);
plt.ylim(0, N)
plt.title(f"Polymer Chain (L={L})")
plt.gca().set_aspect('equal')
plt.show()
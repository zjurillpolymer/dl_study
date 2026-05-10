"""
分析 EGNN 预测结果 — 可视化 + 等变性验证
"""
import warnings
warnings.filterwarnings('ignore')

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader

from egnn import EGNN, batch_fully_connected_edges, create_fully_connected_edges

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, '..', 'data')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 加载数据 ──
dataset = MD17(root=DATA_ROOT, name='ethanol')
N_TRAIN, N_VAL, N_TEST = 5000, 1000, 500

# 统计数据（和 train.py 一致）
energy_stats = torch.tensor([dataset[i].energy.item() for i in range(N_TRAIN)])
force_stats = torch.cat([dataset[i].force for i in range(N_TRAIN)])
energy_mean, energy_std = energy_stats.mean(), energy_stats.std()
force_mean, force_std = force_stats.mean(), force_stats.std()

# 测试集
test_loader = DataLoader(
    dataset[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST],
    batch_size=16, shuffle=False)

# ── 加载模型 ──
model = EGNN(num_atom_types=10, hidden_dim=128, n_layers=4).to(device)
model.load_state_dict(torch.load(
    os.path.join(SCRIPT_DIR, 'egnn_best.pt'), map_location=device, weights_only=True))
model.eval()

# ── 预测 ──
all_energy_pred, all_energy_true = [], []
all_force_pred, all_force_true = [], []

def normalize(batch):
    e = (batch.energy - energy_mean.to(device)) / energy_std.to(device)
    f = (batch.force - force_mean.to(device)) / force_std.to(device)
    return e, f

for batch in test_loader:
    batch = batch.to(device)
    edge_index = batch_fully_connected_edges(batch.ptr, device)
    e_pred, f_pred = model.energy_and_forces(
        batch.z, batch.pos, edge_index, batch.batch, create_graph=False)
    _, f_true = normalize(batch)
    e_true = batch.energy

    all_energy_pred.append(e_pred.cpu())
    all_energy_true.append(e_true.cpu())
    all_force_pred.append(f_pred.cpu())
    all_force_true.append(f_true.cpu())

energy_pred = torch.cat(all_energy_pred)
energy_true = torch.cat(all_energy_true)
force_pred = torch.cat(all_force_pred)
force_true = torch.cat(all_force_true)

# 转换为有物理单位的数值
energy_pred_phys = energy_pred * energy_std + energy_mean
energy_true_phys = energy_true

energy_rmse = (energy_pred_phys - energy_true_phys).pow(2).mean().sqrt().item()
force_pred_phys = force_pred * force_std + force_mean
force_rmse = (force_pred_phys - force_true).pow(2).mean().sqrt().item()

print(f"测试集结果 (标准化空间):")
print(f"  能量 RMSE (std): {energy_rmse:.6f} Hartree")
print(f"  力   RMSE (std): {force_rmse:.6f} Hartree/Bohr")

# ── 1. 能量散点图 ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.scatter(energy_true_phys, energy_pred_phys, s=8, alpha=0.5, c='#1f77b4')
lim = [energy_true_phys.min(), energy_true_phys.max()]
ax.plot(lim, lim, 'k--', lw=1, label='Perfect')
ax.set_xlabel('True Energy (Hartree)')
ax.set_ylabel('Predicted Energy (Hartree)')
ax.set_title(f'Energy Prediction\nRMSE = {energy_rmse:.6f}')
ax.legend()
ax.set_aspect('equal')

# ── 2. 力散点图 ──
ax = axes[1]
ax.scatter(force_true.numpy().flatten(), force_pred.numpy().flatten(),
           s=3, alpha=0.3, c='#ff7f0e')
lim = [force_true.min(), force_true.max()]
ax.plot(lim, lim, 'k--', lw=1, label='Perfect')
ax.set_xlabel('True Force (normalized)')
ax.set_ylabel('Predicted Force (normalized)')
ax.set_title(f'Force Prediction\nRMSE = {force_rmse:.4f}')
ax.legend()
ax.set_aspect('equal')

# ── 3. 能量沿轨迹对比 ──
ax = axes[2]
n_show = min(16, len(energy_true))
ax.plot(range(n_show), energy_true_phys[:n_show], 'o-', label='True', ms=4)
ax.plot(range(n_show), energy_pred_phys[:n_show], 's--', label='Pred', ms=4)
ax.set_xlabel('Conformation index')
ax.set_ylabel('Energy (Hartree)')
ax.set_title('Energy along trajectory (first 16)')
ax.legend()

plt.tight_layout()
save_path = os.path.join(SCRIPT_DIR, 'egnn_results.png')
plt.savefig(save_path, dpi=150)
print(f"\n结果已保存: {save_path}")

# ── 等变性验证 ──
print("\n── 等变性验证 ──")
sample = dataset[N_TRAIN + N_VAL]  # 取一个测试样本
sample = sample.to(device)
ei = create_fully_connected_edges(sample.num_nodes).to(device)

e_orig = model(sample.z, sample.pos, ei, None)
e_orig_phys = e_orig.item() * energy_std + energy_mean

# 随机旋转
rotation = torch.eye(3).to(device)
theta = 0.5
c, s = np.cos(theta), np.sin(theta)
rotation[0, 0] = c; rotation[0, 1] = -s
rotation[1, 0] = s; rotation[1, 1] = c

pos_rotated = sample.pos @ rotation.T
e_rot = model(sample.z, pos_rotated, ei, None)
e_rot_phys = e_rot.item() * energy_std + energy_mean

print(f"原始能量: {e_orig_phys:.6f} Hartree")
print(f"旋转后能量: {e_rot_phys:.6f} Hartree")
diff = abs(e_orig_phys - e_rot_phys)
print(f"差值: {diff:.8f} Hartree")
print(f"等变性: {'✓ 通过' if diff < 1e-4 else '✗ 不通过'}")

# 平移不变性
pos_shifted = sample.pos + torch.tensor([1.0, 2.0, 3.0]).to(device)
e_shift = model(sample.z, pos_shifted, ei, None)
e_shift_phys = e_shift.item() * energy_std + energy_mean
diff_shift = abs(e_orig_phys - e_shift_phys)
print(f"\n平移后能量: {e_shift_phys:.6f} Hartree")
print(f"差值: {diff_shift:.8f} Hartree")
print(f"平移不变性: {'✓ 通过' if diff_shift < 1e-4 else '✗ 不通过'}")

print(f"\n可视化文件: {save_path}")

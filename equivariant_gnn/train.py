"""
在 MD17 Ethanol 上训练等变消息传递网络预测能量+力

架构：RBF 距离展开 + 消息传递（非残差）+ 求和输出
归一化：训练循环中动态计算，不修改 Data 对象
"""
import warnings
warnings.filterwarnings('ignore')

import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader

from egnn import EGNN, batch_fully_connected_edges

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# ── 数据加载 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, '..', 'data')

print("加载 MD17 Ethanol...")
dataset = MD17(root=DATA_ROOT, name='ethanol')
print(f"总构象数: {len(dataset)}")

N_TRAIN, N_VAL, N_TEST = 5000, 1000, 500

# 统计数据
energy_stats = torch.tensor([dataset[i].energy.item() for i in range(N_TRAIN)])
force_stats = torch.cat([dataset[i].force for i in range(N_TRAIN)])
energy_mean, energy_std = energy_stats.mean(), energy_stats.std()
force_mean, force_std = force_stats.mean(), force_stats.std()
print(f"能量: mean={energy_mean:.4f}, std={energy_std:.4f}")
print(f"力:   mean={force_mean:.4f}, std={force_std:.4f}")

train_loader = DataLoader(dataset[:N_TRAIN], batch_size=16, shuffle=True)
val_loader   = DataLoader(dataset[N_TRAIN:N_TRAIN + N_VAL], batch_size=16, shuffle=False)
test_loader  = DataLoader(dataset[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST], batch_size=16, shuffle=False)

# ── 模型 ──
model = EGNN(num_atom_types=10, hidden_dim=128, n_layers=4).to(device)
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


def normalize(batch):
    e = (batch.energy - energy_mean.to(device)) / energy_std.to(device)
    f = (batch.force - force_mean.to(device)) / force_std.to(device)
    return e, f


def train_epoch():
    model.train()
    total_e, total_f = 0, 0
    for batch in train_loader:
        batch = batch.to(device)
        edge_index = batch_fully_connected_edges(batch.ptr, device)
        optimizer.zero_grad()

        e_pred, f_pred = model.energy_and_forces(
            batch.z, batch.pos, edge_index, batch.batch)
        e_tgt, f_tgt = normalize(batch)

        loss_e = F.mse_loss(e_pred, e_tgt)
        loss_f = F.mse_loss(f_pred, f_tgt)
        loss = loss_e + loss_f * 0.5

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_e += loss_e.item() * batch.num_graphs
        total_f += loss_f.item() * batch.num_graphs

    n = len(train_loader.dataset)
    return total_e / n, total_f / n


@torch.no_grad()
def eval_epoch():
    model.eval()
    total = 0
    for batch in val_loader:
        batch = batch.to(device)
        edge_index = batch_fully_connected_edges(batch.ptr, device)
        e_pred = model(batch.z, batch.pos, edge_index, batch.batch)
        e_tgt, _ = normalize(batch)
        total += F.mse_loss(e_pred, e_tgt).item() * batch.num_graphs
    return total / len(val_loader.dataset)


# ── 训练 ──
print(f"\n{'Epoch':>6} | {'E Loss':>10} | {'F Loss':>10} | {'Val Loss':>10} | {'Time':>8}")
print("-" * 52)

best_val = float('inf')
for epoch in range(1, 51):
    t0 = time.time()
    loss_e, loss_f = train_epoch()
    val_loss = eval_epoch()
    scheduler.step()
    elapsed = time.time() - t0

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, 'egnn_best.pt'))

    if epoch == 1 or epoch % 5 == 0:
        print(f"{epoch:6d} | {loss_e:10.6f} | {loss_f:10.6f} | {val_loss:10.6f} | {elapsed:6.1f}s")

# ── 测试 ──
model.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'egnn_best.pt'), weights_only=True))
model.eval()

energy_mae = 0
force_mae = 0
n_test = 0
for batch in test_loader:
    batch = batch.to(device)
    edge_index = batch_fully_connected_edges(batch.ptr, device)

    with torch.no_grad():
        energy_pred = model(batch.z, batch.pos, edge_index, batch.batch)
    _, forces_pred = model.energy_and_forces(
        batch.z, batch.pos, edge_index, batch.batch, create_graph=False)

    energy_true = batch.energy
    forces_true = batch.force

    energy_mae += (energy_pred * energy_std + energy_mean - energy_true).abs().sum().item()
    force_mae += (forces_pred * force_std + force_mean - forces_true).abs().sum().item()
    n_test += batch.num_graphs

energy_mae /= n_test
n_atoms_total = sum(dataset[N_TRAIN + N_VAL + i].num_nodes for i in range(N_TEST))
force_mae /= n_atoms_total * 3

print(f"\n{'='*50}")
print("测试集结果")
print(f"{'='*50}")
print(f"  能量 MAE: {energy_mae:.6f} Hartree")
print(f"  力   MAE: {force_mae:.6f} Hartree/Bohr")
print(f"{'='*50}")

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

# ── 4080 优化 ──
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch._dynamo.config.capture_scalar_outputs = True

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
print(f"每分子力分量数: {force_stats.shape[0] // N_TRAIN} × 3 = {force_stats.shape[-1]}")

train_loader = DataLoader(
    dataset[:N_TRAIN], batch_size=256, shuffle=True,
    num_workers=4, pin_memory=True)
val_loader   = DataLoader(
    dataset[N_TRAIN:N_TRAIN + N_VAL], batch_size=256, shuffle=False,
    num_workers=4, pin_memory=True)
test_loader  = DataLoader(
    dataset[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST], batch_size=256, shuffle=False,
    num_workers=4, pin_memory=True)

# ── 模型 ──
model = EGNN(num_atom_types=10, hidden_dim=128, n_layers=6).to(device)
model = torch.compile(model)
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)
scaler = torch.amp.GradScaler()


def normalize(batch):
    e = (batch.energy - energy_mean.to(device)) / energy_std.to(device)
    f = (batch.force - force_mean.to(device)) / force_std.to(device)
    return e, f


def train_epoch():
    model.train()
    total_e, total_f = 0, 0
    for batch in train_loader:
        batch = batch.to(device, non_blocking=True)
        edge_index = batch_fully_connected_edges(batch.ptr, device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            e_pred, f_pred = model.energy_and_forces(
                batch.z, batch.pos, edge_index, batch.batch)
            e_tgt, f_tgt = normalize(batch)
            loss_e = F.mse_loss(e_pred, e_tgt)
            loss_f = F.mse_loss(f_pred, f_tgt)
            loss = loss_e + loss_f * 1.0

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        total_e += loss_e.item() * batch.num_graphs
        total_f += loss_f.item() * batch.num_graphs

    n = len(train_loader.dataset)
    return total_e / n, total_f / n


def print_epoch(i, e, f, v, t):
    print(f"{i:6d} | {e:10.4f} | {f:10.4f} | {v:10.4f} | {t:6.1f}s")
    if i == 1:
        print(f"  (能量 MAE ≈ {e**0.5 * energy_std * (2/3.14159)**0.5:.2f}, 力 MAE ≈ {f**0.5 * force_std * (2/3.14159)**0.5:.2f})")


@torch.no_grad()
def eval_epoch():
    model.eval()
    total = 0
    for batch in val_loader:
        batch = batch.to(device, non_blocking=True)
        edge_index = batch_fully_connected_edges(batch.ptr, device)
        with torch.amp.autocast(device_type='cuda'):
            e_pred = model(batch.z, batch.pos, edge_index, batch.batch)
        e_tgt, _ = normalize(batch)
        total += F.mse_loss(e_pred, e_tgt).item() * batch.num_graphs
    return total / len(val_loader.dataset)


# ── 训练 ──
print(f"\n{'Epoch':>6} | {'E Loss':>10} | {'F Loss':>10} | {'Val Loss':>10} | {'Time':>8}")
print("-" * 52)

best_val = float('inf')
for epoch in range(1, 301):
    t0 = time.time()
    loss_e, loss_f = train_epoch()
    val_loss = eval_epoch()
    cos_scheduler.step()
    plateau_scheduler.step(val_loss)
    elapsed = time.time() - t0

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, 'egnn_best.pt'))

    if epoch == 1 or epoch % 5 == 0:
        print_epoch(epoch, loss_e, loss_f, val_loss, elapsed)

# ── 测试 ──
model.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'egnn_best.pt'), weights_only=True))
model.eval()
torch.compiler.reset()  # 强制重新编译，排除缓存问题

energy_mae = 0
force_mae = 0
n_test = 0
for batch in test_loader:
    batch = batch.to(device, non_blocking=True)
    edge_index = batch_fully_connected_edges(batch.ptr, device)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        energy_pred = model(batch.z, batch.pos, edge_index, batch.batch)

    # 诊断：检查各变量的 dtype、device、是否含 inf/nan
    print(f"\n--- 诊断 ---")
    print(f"energy_pred: shape={energy_pred.shape}, dtype={energy_pred.dtype}, device={energy_pred.device}, "
          f"inf={torch.isinf(energy_pred).any()}, nan={torch.isnan(energy_pred).any()}, "
          f"min={energy_pred.min():.4f}, max={energy_pred.max():.4f}")
    print(f"energy_std:  dtype={energy_std.dtype}, device={energy_std.device}")
    print(f"energy_mean: dtype={energy_mean.dtype}, device={energy_mean.device}")
    print(f"energy_true: shape={batch.energy.shape}, dtype={batch.energy.dtype}, device={batch.energy.device}, "
          f"inf={torch.isinf(batch.energy).any()}, nan={torch.isnan(batch.energy).any()}, "
          f"min={batch.energy.min():.4f}, max={batch.energy.max():.4f}")

    energy_diff = (energy_pred * energy_std.to(device) + energy_mean.to(device) - batch.energy)
    print(f"energy_diff: "
          f"inf={torch.isinf(energy_diff).any()}, nan={torch.isnan(energy_diff).any()}, "
          f"min={energy_diff.min():.4f}, max={energy_diff.max():.4f}")
    # 如果有 inf，看一下哪些位置
    if torch.isinf(energy_diff).any():
        inf_idx = torch.where(torch.isinf(energy_diff))
        print(f"    inf at indices: {inf_idx}")
        print(f"    energy_pred at those: {energy_pred[inf_idx]}")
        print(f"    energy_true at those: {batch.energy[inf_idx]}")
    print(f"--- 诊断结束 ---\n")

    pos = batch.pos.clone().detach().requires_grad_(True)
    with torch.amp.autocast(device_type='cuda'):
        energy_grad = model(batch.z, pos, edge_index, batch.batch)
    forces_pred = -torch.autograd.grad(energy_grad.sum(), pos, create_graph=False)[0]

    if torch.isinf(energy_grad).any() or torch.isnan(energy_grad).any():
        print(f"    *** energy_grad (for forces) also has inf/nan!")
    else:
        print(f"    energy_grad OK: min={energy_grad.min():.4f}, max={energy_grad.max():.4f}")

    energy_true = batch.energy
    forces_true = batch.force

    energy_mae += (energy_pred * energy_std.to(device) + energy_mean.to(device) - energy_true).abs().sum().item()
    force_mae += (forces_pred * force_std + force_mean - forces_true).abs().sum().item()
    n_test += batch.num_graphs

energy_mae /= n_test
n_atoms_total = sum(dataset[N_TRAIN + N_VAL + i].num_nodes for i in range(N_TEST))
force_mae /= n_atoms_total * 3

print(f"\n{'='*50}")
print("测试集结果")
print(f"{'='*50}")
print(f"  能量 MAE: {energy_mae:.4f} kcal/mol")
print(f"  力   MAE: {force_mae:.4f} kcal/(mol·Å)")
print(f"{'='*50}")

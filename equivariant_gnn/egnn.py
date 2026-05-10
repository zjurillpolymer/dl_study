"""
SchNet 风格等变消息传递网络
- 连续滤波卷积: m_ij = h_j ⊙ W(RBF(||xi-xj||))
- Shifted Softplus 激活
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_fully_connected_edges(n):
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
    mask = i != j
    return torch.stack([i[mask], j[mask]], dim=0)


def batch_fully_connected_edges(batch_ptr, device='cpu'):
    edge_indices = []
    for g in range(len(batch_ptr) - 1):
        start = batch_ptr[g]
        end = batch_ptr[g + 1]
        n_atoms = end - start
        ei = create_fully_connected_edges(n_atoms)
        ei = ei + start
        edge_indices.append(ei)
    return torch.cat(edge_indices, dim=1).to(device)


class ShiftedSoftplus(nn.Module):
    """SchNet 使用的激活函数: ln(0.5 + 0.5*exp(x))，比 ReLU 更平滑"""
    def forward(self, x):
        return F.softplus(x) - 0.5  # shifted so f(0) = ln(2) - 0.5 ≈ 0.19


class RBFExpansion(nn.Module):
    """高斯径向基函数展开"""
    def __init__(self, n_rbf=32, cutoff=4.0):
        super().__init__()
        centers = torch.linspace(0, cutoff, n_rbf)
        self.register_buffer('centers', centers)
        self.gamma = 0.5 / (cutoff / n_rbf) ** 2

    def forward(self, dist):
        d = dist
        c = self.centers.unsqueeze(0)
        return torch.exp(-self.gamma * (d - c) ** 2)


class SchNetConv(nn.Module):
    """连续滤波卷积层（SchNet 核心）"""

    def __init__(self, hidden_dim, rbf_dim=32):
        super().__init__()
        # 连续滤波: RBF(d) → [hidden_dim]（只依赖距离）
        self.filter_net = nn.Sequential(
            nn.Linear(rbf_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 更新网络: Σ m_ij → 新特征
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, x, edge_index, rbf):
        n = h.size(0)
        i, j = edge_index

        rel_pos = x[i] - x[j]
        dist = rel_pos.norm(dim=-1, keepdim=True)
        rbf_feat = rbf(dist)

        # 连续滤波: W = f(RBF(d))，和源特征逐元素乘
        W = self.filter_net(rbf_feat)
        m_ij = h[j] * W  # [n_edges, hidden_dim]

        # 聚合（对目标节点求和）
        aggr = torch.zeros(n, h.size(1), device=h.device)
        aggr = aggr.index_add(0, i, m_ij)
        deg = torch.zeros(n, device=h.device)
        deg = deg.index_add(0, i, torch.ones(m_ij.size(0), device=h.device))
        aggr = aggr / deg.clamp(min=1).view(-1, 1)

        h_new = self.update_net(aggr)
        return h_new, x


class OutputBlock(nn.Module):
    """逐原子 MLP + 求和输出能量"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim), ShiftedSoftplus(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, batch=None):
        E_atom = self.net(h).squeeze(-1)
        if batch is not None:
            n_graphs = batch.max().item() + 1
            energy = torch.zeros(n_graphs, device=h.device)
            energy = energy.index_add(0, batch, E_atom)
        else:
            energy = E_atom.sum()
        return energy


class EGNN(nn.Module):
    """SchNet 风格网络 — 预测分子能量 + 力"""

    def __init__(self, num_atom_types=10, hidden_dim=128, n_layers=6):
        super().__init__()
        self.atom_embed = nn.Embedding(num_atom_types + 1, hidden_dim)
        self.rbf = RBFExpansion(n_rbf=32, cutoff=4.0)
        self.convs = nn.ModuleList([
            SchNetConv(hidden_dim) for _ in range(n_layers)
        ])
        self.output = OutputBlock(hidden_dim)

    def forward(self, z, pos, edge_index, batch=None):
        h = self.atom_embed(z)
        for conv in self.convs:
            h, pos = conv(h, pos, edge_index, self.rbf)
        return self.output(h, batch)

    def energy_and_forces(self, z, pos, edge_index, batch=None, create_graph=True):
        pos.requires_grad_(True)
        energy = self.forward(z, pos, edge_index, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        if not create_graph:
            return energy.detach(), forces.detach()
        return energy, forces

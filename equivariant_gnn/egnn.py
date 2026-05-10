"""
分子势能预测 — 等变消息传递网络

核心设计：
  - 边特征 = RBF 展开 ||xi - xj||（旋转不变）
  - 消息 φ_m(hi, hj, RBF(d)) → 连续滤波卷积
  - 节点更新 h_new = h + φ_h([h, Σm_ij])（残差）
  - 能量 = Σ φ_out(h_i)（求和不变）
  - 力 = -∇_x E（autograd）
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


class RBFExpansion(nn.Module):
    """高斯径向基函数展开: ||xi - xj|| → [exp(-γ(d - μ_k)²)]"""

    def __init__(self, n_rbf=32, cutoff=10.0):
        super().__init__()
        self.n_rbf = n_rbf
        centers = torch.linspace(0, cutoff, n_rbf)
        self.register_buffer('centers', centers)
        self.gamma = 1.0 / (cutoff / n_rbf)

    def forward(self, distances):
        d = distances  # [n, 1] already
        c = self.centers.unsqueeze(0)  # [1, n_rbf]
        return torch.exp(-self.gamma * (d - c) ** 2)  # [n, n_rbf]


class MessageLayer(nn.Module):
    """消息传递层（残差连接）"""

    def __init__(self, hidden_dim, rbf_dim=32):
        super().__init__()
        # 消息网络: [h_i, h_j, RBF(d)] → m_ij
        self.msg_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + rbf_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 节点更新: [h, Σm] → Δh（残差）
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, x, edge_index, rbf):
        n = h.size(0)
        i, j = edge_index

        rel_pos = x[i] - x[j]
        dist = rel_pos.norm(dim=-1, keepdim=True)  # [n_edges, 1]
        rbf_feat = rbf(dist)  # [n_edges, rbf_dim]

        m_ij = self.msg_net(torch.cat([h[i], h[j], rbf_feat], dim=-1))

        deg = torch.zeros(n, device=h.device)
        deg = deg.index_add(0, i, torch.ones(m_ij.size(0), device=h.device))
        aggr = torch.zeros_like(h)
        aggr = aggr.index_add(0, i, m_ij) / deg.clamp(min=1).view(-1, 1)

        h_new = self.update_net(torch.cat([h, aggr], dim=-1))
        return h_new, x


class EGNN(nn.Module):
    """等变消息传递网络 — 预测分子能量 + 力"""

    def __init__(self, num_atom_types=10, hidden_dim=128, n_layers=4):
        super().__init__()
        self.atom_embed = nn.Embedding(num_atom_types + 1, hidden_dim)
        self.rbf = RBFExpansion(n_rbf=32, cutoff=10.0)
        self.layers = nn.ModuleList([
            MessageLayer(hidden_dim) for _ in range(n_layers)
        ])
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, pos, edge_index, batch=None):
        h = self.atom_embed(z)
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, self.rbf)

        E_atom = self.output_net(h).squeeze(-1)

        if batch is not None:
            n_graphs = batch.max().item() + 1
            energy = torch.zeros(n_graphs, device=h.device)
            energy = energy.index_add(0, batch, E_atom)
        else:
            energy = E_atom.sum()
        return energy

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

"""
GNN 入门 —— 从经典图论到消息传递
================================
从 graph_exercise.py 的遍历/最短路/拓扑排序
"跃迁"到图神经网络的核心思想。

核心转变：
  经典图论：人为规则遍历图（BFS/DFS/Dijkstra）
  GNN：    让图自己学习 → 消息传递 (Message Passing)

  即：每个节点"看完"邻居后更新自己。
  BFS 是"广播"，消息传递是"带着权重学习地广播"。
"""

import os
# 避免 macOS 上 PyTorch + sklearn 的 OpenBLAS/OMP 线程冲突
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# 路径基于脚本自身位置，不依赖工作目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ============================================================
# 1. 从 BFS 到消息传递 —— 直观理解
# ============================================================
# BFS 中，节点从邻居收集信息：
#   v 的下一层 = 所有邻居 u
#   但所有邻居一视同仁，没有"权重"概念。
#
# GNN 消息传递 (Message Passing) 三步：
#   ① Message：每个邻居发消息给中心节点
#   ② Aggregate：中心节点汇总邻居消息（求和/平均/注意力）
#   ③ Update：中心节点用自己的特征 + 汇总消息更新自己
#
# 公式：
#   h_v^{(l+1)} = σ( W · AGG({ h_u^{(l)} : u ∈ N(v) }) + B · h_v^{(l)} )
#
# 对比经典图论：
#   Dijkstra 的松弛操作 dist[v] = min(dist[v], dist[u] + w)
#   本质上也是一种"消息传递"——只不过消息是距离，AGG 是取 min
#   GNN 把这个模式推广到了"可学习的特征传递"


# ============================================================
# 2. 最简单的 GNN 层 —— 从零实现
# ============================================================

class SimpleMessagePassing(nn.Module):
    """
    最简单的消息传递层 —— 求和聚合 + 线性变换

    对应公式：h_v' = W · sum({h_u : u ∈ N(v)})
    （简化版，没有自连接和偏置）

    类比 Dijkstra：dist[v] = min(dist[u] + w)  → 聚合方式是 min
    这里：          h_v'    = W · sum(h_u)      → 聚合方式是 sum（可学习）
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adj):
        """
        x:   [num_nodes, in_features]  节点特征
        adj: [num_nodes, num_nodes]    邻接矩阵（0/1 或 带权）
        """
        # Message: 每个节点把特征发给邻居
        # Aggregate: 对每个节点，求和所有邻居的特征
        # adj @ x 的语义：
        #   result[i] = sum_{j ∈ N(i)} x[j]
        #   这正好是"从邻居聚合信息"
        messages = adj @ x             # [num_nodes, in_features]

        # Update: 对聚合结果做线性变换
        return self.linear(messages)   # [num_nodes, out_features]


def demo_simple_message_passing():
    """用上一步的图来演示消息传递"""
    print("=" * 60)
    print("简单消息传递演示")
    print("=" * 60)

    # 从 graph_exercise.py 的图：A-B-C-D 链
    # A - B - C - D
    # 邻接矩阵 (无权无向)
    adj = torch.tensor([
        [0, 1, 0, 0],   # A
        [1, 0, 1, 0],   # B
        [0, 1, 0, 1],   # C
        [0, 0, 1, 0],   # D
    ], dtype=torch.float32)

    # 给每个节点一个随机特征（比如 2 维）
    torch.manual_seed(42)
    x = torch.randn(4, 2)

    layer = SimpleMessagePassing(2, 3)
    out = layer(x, adj)

    print(f"输入特征:\n{x}")
    print(f"\n邻接矩阵:\n{adj}")
    print(f"\n消息传递后:\n{out}")
    print("  → 每个节点聚合了邻居的特征并做了线性变换")
    print("  → A 只看到 B, B 看到 A+C, C 看到 B+D, D 只看到 C")


# ============================================================
# 3. GCN (Graph Convolutional Network) —— 标准实现
# ============================================================

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network 层
    (Kipf & Welling, ICLR 2017)

    核心改进：
      1. 加入自连接 —— 节点聚合时也考虑自己的特征
      2. 对称归一化 —— 防止度大的节点特征过大
      3. 可选的偏置和激活函数

    公式：
      H' = σ( D^{-1/2} (A + I) D^{-1/2} · H · W )
            \___________  __________/
                       \/
                归一化邻接矩阵

    对比 Kruskal：关注的是"边权排序"
    GCN：关注的也是"边的权重"但权重是可学习的
    """

    def __init__(self, in_features, out_features, bias=True, activation=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation

    def forward(self, x, adj):
        """
        x:   [num_nodes, in_features]
        adj: [num_nodes, num_nodes]  已经对称归一化的邻接矩阵（含自连接）
        """
        # Message + Aggregate: 归一化邻接矩阵 @ 特征
        x = adj @ x          # [num_nodes, in_features]
        # Update: 线性变换
        x = self.linear(x)   # [num_nodes, out_features]
        if self.activation:
            x = F.relu(x)
        return x


def normalize_adj(adj):
    """
    对称归一化邻接矩阵 —— 添加自连接并归一化

    D^{-1/2} (A + I) D^{-1/2}

    直觉：度大的节点"稀释"邻居的影响，度小的节点"放大"邻居的影响
    就像：人多的地方每个人发言权被稀释，人少的地方每个人发言权更大
    """
    adj = adj + torch.eye(adj.size(0))        # A + I（加自连接）
    deg = adj.sum(dim=1)                       # 度
    deg_inv_sqrt = torch.pow(deg, -0.5)        # D^{-1/2}
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
    # D^{-1/2} @ (A+I) @ D^{-1/2}
    return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]


# ============================================================
# 4. GAT (Graph Attention Network) —— 动态学边权
# ============================================================
# GCN 的缺陷：所有邻居的权重由"度数"决定，是固定的。
#   A 有 2 个邻居，每个邻居权重 = 1/2
#   B 有 100 个邻居，每个邻居权重 = 1/100
#   但事实上，这 100 个邻居对 B 的重要性可能完全不同。
#
# GAT 的核心改进 —— 自注意力 (Self-Attention)：
#   每个节点自己学会"谁的发言更重要"
#
#   公式：
#     ① 算分：  e_ij = LeakyReLU( a^T · [W h_i || W h_j] )
#     ② 归一化：α_ij = softmax_j(e_ij)  只对邻居做 softmax
#     ③ 聚合：  h_i' = σ( Σ α_ij · W h_j )
#
#     其中 a 是可学习的"注意力向量"（参数）
#
#   对比拓扑排序 Kahn 算法：
#     Kahn 用"入度=0"决定处理顺序 —— 这是硬规则
#     GAT 用"注意力分数高"决定谁的贡献大 —— 这是软权重，从数据中学
#
# GAT 还引入了多头注意力 (Multi-head Attention)：
#   多个注意力头独立计算，结果拼接/平均
#   类比：一个头关注"结构相似"，另一个关注"特征相似"


class GATLayer(nn.Module):
    """
    图注意力层 (Graph Attention Layer)
    (Veličković et al., ICLR 2018)

    相比 GCN 的两个关键区别：
      - 每个邻居的权重是动态计算的（不是固定的归一化系数）
      - 同一个节点在不同层中可以有不同"关注重点"
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        """
        in_features:  输入特征维度
        out_features: 每个注意力头的输出维度
        dropout:      注意力 dropout
        alpha:        LeakyReLU 的负斜率
        """
        super().__init__()
        self.dropout = dropout
        self.out_features = out_features

        # 线性变换: W
        self.W = nn.Linear(in_features, out_features, bias=False)
        # 注意力向量: a  (2 * out_features 因为拼接了两个节点的特征)
        self.a = nn.Parameter(torch.empty(2 * out_features))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(self, x, adj):
        """
        x:   [num_nodes, in_features]
        adj: [num_nodes, num_nodes]  邻接矩阵（0/1）

        返回: [num_nodes, out_features]
        """
        n = x.size(0)
        # 1. 线性变换
        Wh = self.W(x)                    # [n, out_features]

        # 2. 计算所有节点对的注意力分数
        #    拼接 (Wh_i || Wh_j) → 用广播计算所有 i, j 对
        Wh_i = Wh.unsqueeze(1)            # [n, 1, out_features]
        Wh_j = Wh.unsqueeze(0)            # [1, n, out_features]
        concat = torch.cat([               # [n, n, 2*out_features]
            Wh_i.expand(n, n, -1),
            Wh_j.expand(n, n, -1)
        ], dim=-1)

        # 3. 用 a 打分 + LeakyReLU
        e = self.leaky_relu(concat @ self.a)  # [n, n, 1] → [n, n]

        # 4. 只保留邻居的分数（mask 掉非邻居）
        e = e.masked_fill(adj == 0, float('-inf'))

        # 5. Softmax 归一化（只对每个节点的邻居做）
        attn = F.softmax(e, dim=1)             # [n, n]

        # 6. Dropout（防止过拟合）
        attn = F.dropout(attn, self.dropout, training=self.training)

        # 7. 加权聚合邻居消息
        h = attn @ Wh                          # [n, out_features]

        return h


class GAT(nn.Module):
    """
    2 层 GAT 用于节点分类

    第一层: 多头注意力 (8 头)，每个头独立参数，拼接输出
    第二层: 单头注意力，输出分类 logits
    """

    def __init__(self, in_features, hidden_features, num_classes,
                 num_heads=8, dropout=0.6, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_features = hidden_features

        # 第一层: num_heads 个独立注意力头
        self.heads = nn.ModuleList([
            GATLayer(in_features, hidden_features, dropout, alpha)
            for _ in range(num_heads)
        ])
        # 第二层: 单头，输出分类
        self.conv2 = GATLayer(hidden_features * num_heads, num_classes,
                              dropout, alpha)

    def forward(self, x, adj):
        # 第一层: 每个头独立计算，拼接
        heads = []
        for head in self.heads:
            h = head(x, adj)
            heads.append(h)
        x = torch.cat(heads, dim=-1)   # [n, hidden_features * num_heads]
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # 第二层: 单头输出
        x = self.conv2(x, adj)
        return x


def train_karate_club_gat():
    """训练 GAT 在空手道俱乐部数据上做节点分类"""
    print("\n" + "=" * 60)
    print("空手道俱乐部 —— GAT 节点分类")
    print("=" * 60)

    adj, features, labels, train_mask = load_karate_club()

    model = GAT(in_features=34, hidden_features=8, num_classes=2,
                num_heads=8, dropout=0.6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=5e-4)

    print(f"训练样本: {train_mask.sum().item()} 个")
    print(f"节点总数: {features.size(0)} 个")
    print(f"注意力头数: 8")
    print("=" * 60)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(features, adj)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            pred = out.argmax(dim=1)
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            test_acc = (pred[~train_mask] == labels[~train_mask]).float().mean()
            print(f"  Epoch {epoch + 1:3d} | Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc:.2f} | Test Acc: {test_acc:.2f}")

    model.eval()
    with torch.no_grad():
        out = model(features, adj)
        pred = out.argmax(dim=1)
        acc = (pred == labels).float().mean()
        print(f"\n最终准确率: {acc:.2%} ({int(pred.sum())} / {len(labels)})")

    return model, features, adj, labels, pred


# ============================================================
# 5. 实战：Zachary's Karate Club —— 节点分类
# ============================================================
# 经典图数据集：一个空手道俱乐部的 34 个成员，
# 因教练 vs 管理员的分裂形成两个社区。
# 我们用 GCN 来学习"每个成员属于哪个社区"。

def load_karate_club():
    """
    加载空手道俱乐部图数据

    返回：
      adj:        [34, 34] 邻接矩阵
      features:   [34, 34] one-hot 特征（每个节点一个 unique 向量）
      labels:     [34] 标签（0 或 1，代表分裂后的两个社区）
      train_mask: [34] 训练集掩码（每个类只标 1 个样本，半监督）
    """
    # 边列表 (来自真实数据)
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
        (0, 10), (0, 11), (0, 12), (0, 13), (0, 16), (0, 17), (0, 19),
        (0, 21), (0, 31), (1, 2), (1, 3), (1, 7), (1, 13), (1, 17),
        (1, 19), (1, 21), (1, 30), (2, 3), (2, 7), (2, 8), (2, 9),
        (2, 13), (2, 27), (2, 28), (2, 32), (3, 7), (3, 12), (3, 13),
        (4, 6), (4, 10), (5, 6), (5, 10), (5, 16), (6, 16), (8, 30),
        (8, 32), (8, 33), (9, 33), (13, 33), (14, 32), (14, 33),
        (15, 32), (15, 33), (18, 32), (18, 33), (19, 33), (20, 32),
        (20, 33), (22, 32), (22, 33), (23, 25), (23, 27), (23, 29),
        (23, 32), (23, 33), (24, 25), (24, 27), (24, 31), (25, 31),
        (26, 29), (26, 33), (27, 33), (28, 31), (28, 33), (29, 32),
        (29, 33), (30, 32), (30, 33), (31, 32), (31, 33), (32, 33),
    ]

    # 构建邻接矩阵
    n = 34
    adj = torch.zeros(n, n)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1

    # 真实标签（分裂后的两个群体）
    labels = torch.tensor([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1,
        0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ])

    # One-hot 节点特征（每个节点一个 unique 编码）
    features = torch.eye(n)

    # 半监督：每类只标记 1 个样本
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[0] = True   # 社区 0 的代表
    train_mask[33] = True  # 社区 1 的代表

    return adj, features, labels, train_mask


class GCN(nn.Module):
    """2 层 GCN 用于节点分类"""

    def __init__(self, in_features, hidden_features, num_classes):
        super().__init__()
        self.conv1 = GCNLayer(in_features, hidden_features)
        self.conv2 = GCNLayer(hidden_features, num_classes, activation=False)

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = self.conv2(x, adj)
        return x


def train_karate_club():
    """训练 GCN 在空手道俱乐部数据上做节点分类"""
    print("\n" + "=" * 60)
    print("空手道俱乐部 —— GCN 节点分类")
    print("=" * 60)

    adj, features, labels, train_mask = load_karate_club()
    adj_norm = normalize_adj(adj)

    model = GCN(in_features=34, hidden_features=16, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)

    print(f"训练样本: {train_mask.sum().item()} 个")
    print(f"节点总数: {features.size(0)} 个")
    print("=" * 60)

    # 训练
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(features, adj_norm)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            pred = out.argmax(dim=1)
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            test_acc = (pred[~train_mask] == labels[~train_mask]).float().mean()
            print(f"  Epoch {epoch + 1:3d} | Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc:.2f} | Test Acc: {test_acc:.2f}")

    # 评估
    model.eval()
    with torch.no_grad():
        out = model(features, adj_norm)
        pred = out.argmax(dim=1)
        acc = (pred == labels).float().mean()
        print(f"\n最终准确率: {acc:.2%} ({int(pred.sum())} / {len(labels)})")

    return model, features, adj, labels, pred


def visualize_embeddings(model, features, adj, labels, pred, model_name="GCN"):
    """用 t-SNE 可视化 GNN 学到的节点嵌入"""
    # 提取第一层输出作为节点嵌入
    with torch.no_grad():
        if hasattr(model, 'heads'):
            # GAT: 拼接所有注意力头的输出
            heads_out = [head(features, adj) for head in model.heads]
            embeddings = torch.cat(heads_out, dim=-1).numpy()
        else:
            # GCN: 直接取第一层
            embeddings = model.conv1(features, adj).numpy()

    # t-SNE 降维到 2D
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e']
    for i in range(emb_2d.shape[0]):
        plt.scatter(emb_2d[i, 0], emb_2d[i, 1],
                    c=colors[labels[i].item()],
                    s=100, edgecolors='black', linewidths=1)
        plt.text(emb_2d[i, 0] + 0.3, emb_2d[i, 1] + 0.3, str(i),
                 fontsize=9)

    plt.title(f"{model_name} Node Embeddings (t-SNE)")
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='社区 0'),
        Patch(facecolor='#ff7f0e', label='社区 1'),
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, f"karate_embeddings_{model_name.lower()}.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n嵌入可视化已保存 -> {save_path}")


# ============================================================
# 6. 运行所有 Demo
# ============================================================
if __name__ == "__main__":
    demo_simple_message_passing()

    print("\n" + "#" * 60)
    print("# GCN 训练")
    print("#" * 60)
    model_gcn, features, adj, labels, pred_gcn = train_karate_club()
    visualize_embeddings(model_gcn, features, adj, labels, pred_gcn,
                         "GCN")

    print("\n" + "#" * 60)
    print("# GAT 训练")
    print("#" * 60)
    model_gat, _, _, _, pred_gat = train_karate_club_gat()

"""用训练好的模型预测新分子"""

import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool
from rdkit import Chem
import molecule_to_graph
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 模型定义（和训练时一致） ──
class Mol_tox_GAT(nn.Module):
    def __init__(self, in_channels=29, edge_dim=6, hidden=64, heads=4, num_tasks=12):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=heads, edge_dim=edge_dim, concat=True)
        self.gat2 = GATConv(hidden * heads, hidden, heads=heads, edge_dim=edge_dim, concat=True)
        self.lin = nn.Linear(hidden * heads, num_tasks)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.lin(x)

# ── 加载模型 ──
model = Mol_tox_GAT().to(device)
model.load_state_dict(torch.load('best_model.pt', map_location=device, weights_only=False))
model.eval()

# ── Tox21 任务名 ──
tasks = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

# ── 测试分子 ──
molecules = [
    ("Ethanol (乙醇)", "CCO", "基本无毒"),
    ("Benzene (苯)", "c1ccccc1", "致癌, 有毒"),
    ("Formaldehyde (甲醛)", "C=O", "毒性强"),
]

print(f"{'='*90}")
print(f"{'Molecule':20s} {'Label':12s}   Toxicity Predictions (probability)")
print(f"{'='*90}")

for name, smiles, desc in molecules:
    mol = Chem.MolFromSmiles(smiles)
    data = molecule_to_graph.mol_to_pyg_graph(mol)
    data = data.to(device)

    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr,
                       torch.zeros(data.num_nodes, dtype=torch.long, device=device))
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    # 标记高风险任务（>0.5）
    high_risk = [tasks[i] for i, p in enumerate(probs) if p > 0.5]

    print(f"{name:20s} {desc:12s}  ", end="")
    for p in probs:
        if p > 0.5:
            print(f"\033[91m{p:.3f}\033[0m  ", end="")   # 红色高亮
        else:
            print(f"{p:.3f}  ", end="")
    print()
    if high_risk:
        print(f"{'':20s} {'':12s}   ⚠ 高风险: {', '.join(high_risk)}")
    else:
        print(f"{'':20s} {'':12s}   ✓ 无高风险")
    print()

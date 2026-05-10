import warnings

from torch import nn

warnings.filterwarnings('ignore')

from rdkit import RDLogger, Chem
RDLogger.logger().setLevel(RDLogger.ERROR)

import os
import pandas as pd
import numpy as np
import gzip
import urllib.request
import molecule_to_graph
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch import optim



data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'tox21')
os.makedirs(data_dir, exist_ok=True)

csv_path = os.path.join(data_dir, 'tox21.csv')
gz_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'

if not os.path.exists(csv_path):
    print("Downloading Tox21 dataset...")
    gz_path = csv_path + '.gz'
    urllib.request.urlretrieve(gz_url, gz_path)
    with gzip.open(gz_path, 'rt') as f:
        with open(csv_path, 'w') as out:
            out.write(f.read())
    os.remove(gz_path)
    print("Done.")

df = pd.read_csv(csv_path)

# First 12 columns are toxicity labels, then mol_id, then smiles
tasks = [c for c in df.columns if c not in ('mol_id', 'smiles')]
# print(f"\n{'='*60}")
# print(f"Tasks ({len(tasks)} toxicity endpoints):")
# for t in tasks:
#     print(f"  - {t}")

# print(f"\nTotal compounds: {len(df)}")
#
# print(f"\nLabel distribution per task:")
# for t in tasks:
#     counts = df[t].value_counts(dropna=False)
#     pos = counts.get(1, 0)
#     neg = counts.get(0, 0)
#     total = pos + neg
#     print(f"  {t:20s}  pos: {pos:4d} ({100*pos/total:5.1f}%)  neg: {neg:4d}")
#
# print(f"\nSample data (first 3 rows):")
# for i in range(3):
#     print(f"  Row {i}: SMILES={df['smiles'].iloc[i]},  labels={df[tasks].iloc[i].values}")

# Random 80/10/10 split
np.random.seed(42)
n = len(df)
idx = np.random.permutation(n)
n_train = int(0.8 * n)
n_valid = int(0.1 * n)
train_idx = idx[:n_train]
valid_idx = idx[n_train:n_train + n_valid]
test_idx = idx[n_train + n_valid:]

# print(f"\nDataset sizes (random 80/10/10):")
# print(f"  Train:   {len(train_idx)}")
# print(f"  Valid:   {len(valid_idx)}")
# print(f"  Test:    {len(test_idx)}")

train_df = df.iloc[train_idx]
valid_df = df.iloc[valid_idx]
test_df = df.iloc[test_idx]
pos_counts = train_df[tasks].sum()
# print(f"\nPositive label counts per task (train set):")
# for i, t in enumerate(tasks):
#     print(f"  {t}: {int(pos_counts[i])} / {len(train_df)} ({100*pos_counts[i]/len(train_df):.1f}%)")







class Tox21Dataset(Dataset):
    def __init__(self, df, tasks, cache_path="graph_cache.pt"):
        self.df = df.reset_index(drop=True)
        self.tasks = tasks
        self.cache_path = cache_path

        if os.path.exists(cache_path):
            print(f"加载缓存: {cache_path}")
            self.data_list = torch.load(cache_path, weights_only=False)
        else:
            print(f"首次构建图，缓存到: {cache_path}")
            self.data_list = self._preprocess_all_graphs()
            torch.save(self.data_list, cache_path)
            print(f"缓存完成！")

    def _preprocess_all_graphs(self):
        data_list=[]
        total=len(self.df)
        fail=0

        for idx in range(total):
            smiles=self.df['smiles'].iloc[idx]
            labels=self.df[self.tasks].iloc[idx].values

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                fail+=1
                continue

            # clean nan → 0, 同时记 mask 忽略缺失标签
            label = np.nan_to_num(labels, nan=0.0).astype(np.float32)
            mask = (~np.isnan(labels)).astype(np.float32)

            data = molecule_to_graph.mol_to_pyg_graph(mol)
            data.y = torch.tensor(label, dtype=torch.float).view(1, -1)
            data.mask = torch.tensor(mask, dtype=torch.float).view(1, -1)

            data_list.append(data)

        if fail:
            print(f"  (跳过 {fail} 个无法解析的分子)")
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx].clone()



train=Tox21Dataset(train_df, tasks, cache_path="graph_cache_train.pt")
valid=Tox21Dataset(valid_df, tasks, cache_path="graph_cache_valid.pt")
test=Tox21Dataset(test_df, tasks, cache_path="graph_cache_test.pt")

# print(train.len()) #6264
# print(test.get(0)) #Data(x=[16, 29], edge_index=[2, 34], edge_attr=[34, 6], y=[12])

'''
Data(x=[16, 29], edge_index=[2, 34], edge_attr=[34, 6], y=[12])
16个原子，每个原子29个特征
无向图（2）34根化学键
34根化学键 每个6个特征
12个任务，二分类
'''



BATCH_SIZE = 32

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)


class Mol_tox_GAT(nn.Module):
    def __init__(self,in_channels=29,edge_dim=6,hidden=64,heads=4,num_tasks=12):
        super().__init__()
        self.gat1 = GATConv(in_channels,hidden,heads=heads,edge_dim=edge_dim,concat=True) #64*4=256
        self.gat2 = GATConv(hidden*heads,hidden,heads=heads,edge_dim=edge_dim,concat=True)
        self.lin=nn.Linear(hidden*heads,num_tasks)

    def forward(self,x,edge_index, edge_attr, batch):
        x=self.gat1(x,edge_index,edge_attr=edge_attr)
        x=F.relu(x)
        x=F.dropout(x,p=0.2,training=self.training)

        x=self.gat2(x,edge_index,edge_attr=edge_attr)
        x=F.relu(x)
        x=F.dropout(x,p=0.2,training=self.training)
        x=global_mean_pool(x,batch)
        out=self.lin(x)
        return out


# ---------- Training ----------
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = Mol_tox_GAT().to(device)

# pos_weight: 每个任务 负例数/正例数，缓解类别不平衡
pos_counts = train_df[tasks].sum()
neg_counts = len(train_df) - pos_counts
pos_weight = torch.tensor((neg_counts / pos_counts).values, dtype=torch.float).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none') #二分类任务的损失函数

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_epoch(loader):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y)          # [batch_size, 12]
        loss = (loss * batch.mask).sum() / batch.mask.sum()  # 忽略 NaN 标签
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(loader):
    model.eval()
    total_loss = 0
    all_out, all_y, all_mask = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y)
        loss = (loss * batch.mask).sum() / batch.mask.sum()
        total_loss += loss.item() * batch.num_graphs
        all_out.append(out.cpu())
        all_y.append(batch.y.cpu())
        all_mask.append(batch.mask.cpu())
    return (total_loss / len(loader.dataset),
            torch.cat(all_out), torch.cat(all_y), torch.cat(all_mask))


print("Start training...")
best_val_loss = float('inf')
for epoch in range(1, 51):
    train_loss = train_epoch(train_loader)
    val_loss, preds, labels, masks = eval_epoch(valid_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')

    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch {epoch:3d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

# Test

model.load_state_dict(torch.load('best_model.pt', weights_only=False))
test_loss, preds, labels, masks = eval_epoch(test_loader)
print(f"\nTest loss: {test_loss:.4f}")

aucs = []
for i, task in enumerate(tasks):
    m = masks[:, i].bool()
    if m.sum() > 1 and labels[:, i][m].unique().numel() > 1:
        auc = roc_auc_score(labels[:, i][m].numpy(), preds[:, i][m].numpy())
        aucs.append(auc)
    else:
        auc = float('nan')
    print(f"  {task:20s} ROC-AUC: {auc:.4f}" if not np.isnan(auc) else f"  {task:20s} ROC-AUC: N/A")
print(f"  {'Average':20s} ROC-AUC: {np.nanmean(aucs):.4f}")

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# === 1. 数据准备 ===
train_data = pd.read_csv('D:/data/kaggle_house_pred_train.csv')
test_data = pd.read_csv('D:/data/kaggle_house_pred_test.csv')

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features.astype(np.float32)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)

# ✅ 只对训练标签取 log（SalePrice > 0，安全）
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float32).reshape(-1, 1)
train_labels_log = torch.log(train_labels)  # 👈 关键：训练目标是 log(price)

# 注意：test_labels 不存在！K折验证时验证集标签来自 train_labels_log 的一部分

# === 2. 模型 ===
in_features = train_features.shape[1]


def get_net():
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )


# === 3. 损失函数 ===
loss = nn.MSELoss()


# ✅ 修正 log_rmse：现在 labels 是原始 price（用于最终评估）
# 但注意：在 k-fold 中，我们传入的是 log 标签？不！我们要统一！
# 更简单：让 log_rmse 接受原始标签，内部做 log
def log_rmse(net, features, labels):
    # labels: 原始 price（如 train_labels）
    preds = net(features)  # 模型输出 log(price)
    clipped_preds = torch.clamp(torch.exp(preds), min=1.0)  # exp 还原为 price，并限制 ≥1
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


# 但更推荐：训练和验证都用 log 标签，评估时再 exp
# 下面采用更清晰的方式：训练目标 = log(price)，评估时 exp 还原

# === 4. 训练函数（使用 log 标签训练）===
def train(net, train_features, train_labels_log, test_features, test_labels_log,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels_log)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)  # y 是 log(price)
            l.backward()
            optimizer.step()

        # 记录训练集上的 log_rmse（需要原始标签）
        with torch.no_grad():
            train_pred = net(train_features)
            train_rmse = torch.sqrt(loss(train_pred, train_labels_log)).item()
            train_ls.append(train_rmse)

            if test_labels_log is not None:
                test_pred = net(test_features)
                test_rmse = torch.sqrt(loss(test_pred, test_labels_log)).item()
                test_ls.append(test_rmse)

    return train_ls, test_ls


# === 5. K折交叉验证（使用 log 标签）===
def get_k_fold_data(k, i, X, y):  # y 现在是 log(price)
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], dim=0)
            y_train = torch.cat([y_train, y_part], dim=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train_log, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        X_train_fold, y_train_fold, X_val_fold, y_val_fold = get_k_fold_data(k, i, X_train, y_train_log)
        net = get_net()
        train_ls, valid_ls = train(net, X_train_fold, y_train_fold,
                                   X_val_fold, y_val_fold,
                                   num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
    return train_l_sum / k, valid_l_sum / k


# === 6. 调用 ===
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.01, 1e-4, 64  # lr 别用 5！
train_l, valid_l = k_fold(k, train_features, train_labels_log, num_epochs, lr, weight_decay, batch_size)

# 注意：这里的 train_l, valid_l 是 MSE(log_pred, log_true) 的 sqrt，即 RMSLE
print(f'{k}-折验证: 平均训练 RMSLE: {float(train_l):.6f}, 平均验证 RMSLE: {float(valid_l):.6f}')

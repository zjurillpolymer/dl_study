import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义带有 Dropout 的模型
# 我们直接使用 nn.Sequential，这是最清晰的工业级写法
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),  # 暂退法 1
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # 暂退法 2
    nn.Linear(256, 10)
)

# 2. 初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# 3. 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 4. 定义优化器并设置权重衰减 (Weight Decay)
# 注意：weight_decay 参数就是 L2 正则化的系数 lambda
learning_rate = 0.1
wd_lambda = 0.001 # 这里的权重衰减系数

optimizer = optim.SGD(
    net.parameters(),
    lr=learning_rate,
    weight_decay=wd_lambda  # 在这里应用权重衰减
)

# --- 模拟一步训练过程 ---

# 模拟输入 (BatchSize=64, 1通道, 28x28)
X = torch.randn(64, 1, 28, 28)
y = torch.randint(0, 10, (64,))

# 切换到训练模式：Dropout 生效
net.train()

# 梯度清零
optimizer.zero_grad()

# 前向传播 (此时 Dropout 正在随机丢弃神经元)
output = net(X)
l = loss_fn(output, y)

# 反向传播 (此时不仅计算 Loss 梯度，Weight Decay 也会影响权重更新)
l.backward()

# 更新权重
optimizer.step()

print(f"训练完成一次迭代。损失值: {l.item():.4f}")
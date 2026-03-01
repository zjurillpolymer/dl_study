import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import Module


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        # 预测模式：直接用全局变量
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 全连接层
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 卷积层
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

        # 训练模式：用当前 batch 的均值和方差
        X_hat = (X - mean) / torch.sqrt(var + eps)

        # 【关键：原地更新】
        # 使用 [:] 确保修改的是传进来的那个张量的内存，而不是创建一个局部变量
        moving_mean[:] = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var[:] = momentum * moving_var + (1.0 - momentum) * var

    Y = gamma * X_hat + beta  # 缩放和平移
    return Y, moving_mean, moving_var


class BatchNorm(nn.Module):
    def __init__(self,num_features,num_dims):
        super(BatchNorm,self).__init__()
        if num_dims==2:
            shape=(1,num_features)
        else:
            shape=(1,num_features,1,1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.register_buffer('moving_mean', torch.zeros(shape))
        self.register_buffer('moving_var', torch.ones(shape))

    def forward(self,X):
        if self.moving_mean.device!=X.device:
            self.moving_mean=self.moving_mean.to(X.device)
            self.moving_var=self.moving_var.to(X.device)
        Y, _, _ = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var,
            eps=1e-5, momentum=0.9
        )
        return Y

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10)
)

lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 设置损失函数和优化器
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)


# 假设你已经定义好了 net, loss, trainer, train_iter, test_iter, num_epochs

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)


def evaluate_accuracy(net, data_iter, device):
    """计算模型在数据集上的准确率"""
    net.eval()  # 切换到评估模式（影响 BN 和 Dropout）
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            correct += (y_hat.argmax(dim=1) == y).sum().item()
            total += y.numel()
    return correct / total

if __name__ == '__main__':
    for epoch in range(num_epochs):
        net.train()  # 切换到训练模式（BN 使用 batch 统计量）
        running_loss = 0.0
        total_samples = 0

        for i, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)

            # 前向传播
            y_hat = net(X)
            l = loss(y_hat, y)

            # 反向传播 + 优化
            trainer.zero_grad()
            l.backward()
            trainer.step()

            # 累计损失和样本数
            running_loss += l.item() * X.size(0)
            total_samples += X.size(0)

        # 计算平均训练损失
        avg_train_loss = running_loss / total_samples
        # 计算测试准确率
        test_acc = evaluate_accuracy(net, test_iter, device)

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Loss: {avg_train_loss:.4f}, '
              f'Test Acc: {test_acc:.4f}')


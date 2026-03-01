import torch
from torch import nn
from d2l import torch as d2l


# ------------------ 定义 VGG ------------------
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(in_channels * 7 * 7, 4096),  # 注意：这里用 in_channels 更安全
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )


# ------------------ 设置设备 ------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------ 构建模型并移至 GPU ------------------
net = vgg(small_conv_arch).to(device)

# ------------------ 数据加载 ------------------
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

# ------------------ 训练配置 ------------------
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)


# ------------------ 训练循环 ------------------
def train_gpu(net, train_iter, test_iter, loss, num_epochs, trainer, device):
    net.to(device)
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)  # (loss_sum, corrects, total_samples)
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)  # 👈 关键：数据移到 GPU
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            metric.add(l.item() * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]

        # 测试集评估
        net.eval()
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter, device=device)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


# 开始训练
if __name__ == '__main__':
    net = vgg(conv_arch)
    train_gpu(net, train_iter, test_iter, loss, num_epochs, trainer, device)
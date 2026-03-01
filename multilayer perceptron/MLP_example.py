import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


# --- 1. 定义工具函数 ---
def relu(X):
    # torch.zeros_like 会自动继承 X 的设备信息（CPU 或 GPU）
    return torch.max(X, torch.zeros_like(X))


def evaluate_accuracy(data_iter, net, device):
    """在指定设备上评估模型准确率"""
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            # 将数据搬移到 GPU
            X, y = X.to(device), y.to(device)
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


# --- 2. 主程序入口 (Windows 运行必须包含在 if __name__ == '__main__': 中) ---
if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'当前运行设备: {device}')

    # 加载数据
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 模型参数
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    # 初始化参数并移动到 GPU
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, device=device, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, device=device, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))

    params = [W1, b1, W2, b2]


    # 定义网络结构
    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(X @ W1 + b1)  # 矩阵乘法: (batch, 784) @ (784, 256)
        return H @ W2 + b2


    # 损失函数与优化器
    # 注意：这里 reduction 默认就是 'mean'，能让梯度更稳定
    loss = nn.CrossEntropyLoss()
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)

    # 记录训练过程
    train_losses, train_accs, test_accs = [], [], []

    print("开始训练...")
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            # 数据搬移到 GPU
            X, y = X.to(device), y.to(device)

            y_hat = net(X)
            l = loss(y_hat, y)

            updater.zero_grad()
            l.backward()
            updater.step()

            train_l_sum += l.item() * y.shape[0]
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        # 每个 epoch 结束后的评估
        test_acc = evaluate_accuracy(test_iter, net, device)

        epoch_loss = train_l_sum / n
        epoch_acc = train_acc_sum / n

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        test_accs.append(test_acc)

        print(f'epoch {epoch + 1}, loss {epoch_loss:.4f}, train acc {epoch_acc:.3f}, test acc {test_acc:.3f}')

    # --- 3. 绘图 ---
    plt.figure(figsize=(10, 4))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='train acc')
    plt.plot(test_accs, label='test acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
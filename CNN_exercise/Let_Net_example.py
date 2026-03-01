import torch
from torch import nn
from d2l import torch as d2l

'''输入的是一张28*28的图片
torch.Size([1, 6, 28, 28])
torch.Size([1, 6, 28, 28])
torch.Size([1, 6, 14, 14])
torch.Size([1, 16, 10, 10])
torch.Size([1, 16, 10, 10])
torch.Size([1, 16, 5, 5])
torch.Size([1, 400])
torch.Size([1, 120])
torch.Size([1, 120])
torch.Size([1, 84])
torch.Size([1, 84])
torch.Size([1, 10])
'''

net=nn.Sequential(
    nn.Conv2d(1,6,kernel_size=5,padding=2), ### 经过此层后，图像大小仍然是28x28。
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2), ### 平均池化，图像变为14*14
    nn.Conv2d(6,16,kernel_size=5), #10*10
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2), #5*5
    nn.Flatten(), #展平为一维向量
    nn.Linear(16*5*5,120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 10),
)

# X=torch.rand(size=(1,1,28,28),dtype=torch.float32)
# for layer in net:
#     X=layer(X)
#     print(X.size())

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

### 模型训练

def evaluate_accuracy_gpu(net, data_iter,device=None):
    if isinstance(net,nn.Module):
        net.eval()
        if not device:
            device = next(net.parameters()).device
    metric=d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X,list):
                X=[x.to(device) for x in X]
            else:
                X=X.to(device)
            y=y.to(device)
            metric.add(d2l.accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

        test_acc = evaluate_accuracy_gpu(net, test_iter)
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

if __name__ == '__main__':
    lr, num_epochs = 0.1, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


'''training on cuda:0
loss 0.481, train acc 0.818, test acc 0.789
46874.5 examples/sec on cuda:0
xavier初始化+sigmoid

training on cuda:0
loss 0.314, train acc 0.885, test acc 0.870
44216.0 examples/sec on cuda:0
kaiming初始化+ReLU

training on cuda:0
loss 0.278, train acc 0.899, test acc 0.878
46440.5 examples/sec on cuda:0
最大汇聚层'''
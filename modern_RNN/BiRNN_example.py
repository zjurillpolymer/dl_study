import torch
import torch.nn as nn


# 定义双向RNN模型
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义双向RNN层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True)

        # 全连接层，输入维度是 2*hidden_size（因为双向）
        self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # 初始化隐藏状态 (num_layers * 2, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 through RNN
        out, _ = self.rnn(x, h0)
        # out shape: (batch_size, seq_len, 2*hidden_size)

        # 只取最后一个时间步的输出
        out = out[:, -1, :]  # shape: (batch_size, 2*hidden_size)

        # 通过全连接层
        out = self.fc(out)
        return out


# 参数设置
input_size = 10  # 输入特征维度
hidden_size = 20  # 隐藏层维度
num_layers = 2  # RNN层数
output_size = 2  # 输出类别数（例如二分类）
seq_len = 5  # 序列长度
batch_size = 3  # 批次大小

# 创建模型实例
model = BiRNN(input_size, hidden_size, num_layers, output_size)

# 创建随机输入数据 (batch_size, seq_len, input_size)
x = torch.randn(batch_size, seq_len, input_size)

# 前向传播
output = model(x)

print("输入形状:", x.shape)
print("输出形状:", output.shape)
print("输出示例:\n", output)
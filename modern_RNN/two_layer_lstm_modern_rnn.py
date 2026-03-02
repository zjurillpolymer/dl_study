import torch
import torch.nn as nn
'''
构建2层LSTM的RNN
'''
class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        # 我们把 4 个门的权重合并在一个大矩阵里，计算效率更高
        # 4 个门分别是：输入门(i), 遗忘门(f), 输出门(o), 候选memory cell(g)
        self.weight_ih = nn.Parameter(torch.randn(input_size, 4 * hidden_size) * 0.01)
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, 4 * hidden_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x, state):
        h_prev,c_prev=state

        gates=x@self.weight_ih+h_prev@self.weight_hh+self.bias

        i_gate,f_gate,o_gate,g_gate=gates.chunk(4,dim=1)

        i=torch.sigmoid(i_gate)
        f=torch.sigmoid(f_gate)
        o=torch.sigmoid(o_gate)
        g=torch.tanh(o_gate)

        c_next=f*c_prev+i*g
        h_next=o*torch.tanh(c_next)

        return h_next,c_next


class DeepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.layer1 = MyLSTMCell(input_size, hidden_size)
        self.layer2 = MyLSTMCell(hidden_size, hidden_size)  # 输入大小变为 hidden_size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (seq_len, batch_size, input_size)
        seq_len, batch_size, _ = x.size()

        # 初始化两层各自的 (h, c) 状态
        h1, c1 = torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)
        h2, c2 = torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)

        for t in range(seq_len):
            x_t = x[t]

            # 第一层计算
            h1, c1 = self.layer1(x_t, (h1, c1))

            # 第二层计算：接收第一层的隐状态作为输入
            h2, c2 = self.layer2(h1, (h2, c2))

        # 使用最后一层、最后一个时间步的隐状态进行预测
        return self.fc(h2)


# 测试运行
model = DeepLSTM(input_size=10, hidden_size=32, output_size=1)
sample_input = torch.randn(5, 3, 10)  # 序列长5, batch为3, 特征为10
output = model(sample_input)
print(f"最终输出形状: {output.shape}")  # (3, 1)
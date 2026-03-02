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


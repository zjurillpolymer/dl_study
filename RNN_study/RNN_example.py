import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from text_pre_process import load_data_time_machine


batch_size,num_steps=32,35

train_iter,vocab=load_data_time_machine(batch_size,num_steps)

num_hiddens=256 ###隐状态的shape
rnn_layer=nn.RNN(len(vocab),num_hiddens) ###即根据上一个隐藏状态和新词计算下一个隐藏状态
state=torch.zeros((1,batch_size,num_hiddens))  ###初始状态，均设为0
print(state.shape)   ### torch.Size([1, 32, 256])

X=torch.rand(size=(num_steps,batch_size,len(vocab)))  ###默认大小是seq_len*batch_size*……
Y,state_new=rnn_layer(X,state) ### 这里的Y是隐藏状态张量，Y[0]的shape是32*256，state_new是最后一个隐藏状态
print(Y[0].shape)


#@save
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)
    '''
    PyTorch 的 nn.RNN 默认期望输入格式是 (num_steps, batch_size, ...)。
    而数据加载器通常给的是 (batch_size, num_steps)。
    '''
    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)###X每一个元素都是每一个num_step对应的one-hot vector
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state) ###Y的形状是(num_steps,batch_size,num_hiddens)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


device=d2l.try_gpu()
net=RNNModel(rnn_layer, vocab_size=len(vocab))
net=net.to(device)

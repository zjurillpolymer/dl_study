import collections
import math
import torch
from torch import nn
from d2l import torch as d2l

'''
第一层 (Bottom Layer)：直接接触原始数据（词向量）。它负责提取初级的、表面的语义（比如：主语是谁，动词是什么）。

第二层 (Top Layer)：它不看原始词向量，它只看第一层的输出。它负责提取更高级、更抽象的逻辑（比如：这句话的语气是什么，复杂的从句关系是什么）。

两层同时运作
'''

class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0,**kwargs):
        super(Seq2SeqEncoder,self).__init__(**kwargs)

        self.embedding=nn.Embedding(vocab_size,embed_size)  ##嵌入层
        self.rnn=nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)  ## 循环神经网络 注意这里的维度从embed_size变成了num_hiddens

    def forward(self, X, *args): ### 输入的X的维度(batch_size, num_steps)
        X=self.embedding(X) ## (batch_size, num_steps,embed_size)
        X=X.permute(1,0,2) ## ( num_steps,batch_size,embed_size)调换顺序，num_steps优先
        output,state=self.rnn(X)

        return output,state
encoder=Seq2SeqEncoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layers=2)
encoder.eval()
X=torch.zeros((4,7),dtype=torch.long)
output,state=encoder(X)
# print(output.shape) ### (7,4,16)
# print(state.shape) ### torch.Size([2, 4, 16])
'''
state 记录的是最后时刻（Time Step 7）每一层的压缩记忆。 * state[0] 是第一层在读完整个句子后的“总结”。

state[1] 是第二层在读完整个句子后的“总结”。
4个句子就是4个batch_size
'''
# print(torch.allclose(output[-1], state[-1])) ### True


class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0,**kwargs):
        super(Seq2SeqDecoder,self).__init__(**kwargs)
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.GRU(embed_size+num_hiddens,num_hiddens,num_layers,dropout=dropout)
        self.dense=nn.Linear(num_hiddens,vocab_size)


    def init_state(self,enc_outputs,**kwargs):
        return enc_outputs[-1]

    def forward(self,X,state):
        X=self.embedding(X).permute(1,0,2) ##现在是(num_steps,batch_size,embed_size)
        cotext=state[-1].repeat(X.shape[0],1,1) ## 上下文变量
        X_and_context=torch.cat((X,cotext),2)
        output,state=self.rnn(X_and_context,state)
        '''
        输出 output：(7, 4, 16)（最后一层每个时间步的输出）。

        输出 state：(2, 4, 16)（所有层最后一个时间步的状态）。
        '''
        output=self.dense(output).permute(1,0,2)
        return output,state
'''
X的维度是(4,7)
'''
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
# print(output.shape)
# print(state.shape)


# def sequence_mask(X,valid_len,value=0):
#     maxlen=X.size(1)
#     mask = torch.arange((maxlen), dtype=torch.float32,
#                         device=X.device)[None, :] < valid_len[:, None]
#     X[~mask] = value
#     return X

def sequence_mask(X, valid_len, value=0):
    batch_size, seq_len = X.shape
    X_out = X.clone()
    time_steps=torch.arange(seq_len,device=X.device) #标准的时间步[0,1,2]
    valid_len_col=valid_len.view(-1,1) ##reshape
    mask=time_steps<valid_len_col
    X_out[~mask]=value
    return X_out


X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(sequence_mask(X, torch.tensor([1, 2])))
import math
import torch
from torch import nn
from d2l import torch as d2l

'''
原始 X：(2, 3, 4) → reshape 后：(2, 3, 2, 2)（拆分成 2 个头，每个头 2 维）
permute 后：(2, 2, 3, 2)（把 “头” 提到第二个维度，每个头对应 3 个单词的 2 维特征）
最终 return：(2*2=4, 3, 2)（合并批次和头，变成 4 个 “虚拟样本”，每个样本 3 个单词的 2 维特征）
'''

def transpose_qkv(X,num_heads):
    #输⼊X的形状: (batch_size，查询或者“键－值”对的个数，num_hiddens)
    X=X.reshape(X.shape[0],X.shape[1],num_heads,-1) #增加一个维度(batch_size，查询或者“键－值”对的个数，num_heads,num_hiddens/num_heads)
    X=X.permute(0,2,1,3)#(batch_size,num_heads,查询或者“键－值”对的个数，num_hiddens/num_heads)
    return X.reshape(-1,X.shape[2],X.shape[3])#(batch_size,查询或者“键－值”对的个数，num_hiddens/num_heads)

def transpose_output(X,num_heads):
     #逆转transpose_qkv的操作
    X=X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X=X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)


class MultiheadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiheadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention=d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v=nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)


    def forward(self,queries,keys,values,valid_lens):
        queries=transpose_qkv(self.W_q(queries), self.num_heads)
        keys=transpose_qkv(self.W_k(keys), self.num_heads)
        values=transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens=torch.repeat_interleave(
                valid_lens,repeats=self.num_heads,dim=0
            )
        output=self.attention(queries,keys,values,valid_lens)

        output_concat=transpose_output(output,self.num_heads)
        return self.W_o(output_concat)
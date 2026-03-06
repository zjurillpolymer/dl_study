import math
import torch
from torch import nn
from d2l import torch as d2l


num_hiddens,num_heads = 100,5
attention = d2l.MultiHeadAttention(
    key_size=num_hiddens,
    query_size=num_hiddens,
    value_size=num_hiddens,
    num_hiddens=num_hiddens,
    num_heads=num_heads,
    dropout=0.5
)
attention.eval()
batch_size,num_queries,valid_lens=2,4,torch.tensor([3,2])
X=torch.ones((batch_size,num_queries,num_hiddens))
# print(attention(X,X,X,valid_lens).shape)

'''realize positional encoding'''
class PositionalEncoding(nn.Module):
    def __init__(self,num_hiddens,dropout,max_len=1000):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.P=torch.zeros((1,max_len,num_hiddens))
        X = (torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) #生成位置索引，从0到max_len-1，并转为列向量
             / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens))


        self.P[:,:, 0::2] = torch.sin(X)
        self.P[:,:, 1::2] = torch.cos(X)


    def forward(self,X):
        '''
        X 通常是词嵌入向量（Word Embeddings），形状为 (batch_size, seq_len, num_hiddens)。
        '''
        X=X+self.P[:,:X.shape[1],:].to(X.device)
        return self.dropout(X)


'''realize a learnable position encoding function'''



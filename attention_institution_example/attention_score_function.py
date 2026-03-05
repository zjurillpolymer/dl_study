import math
import torch
from torch import nn
from d2l import torch as d2l
from attention_visualization import show_heatmaps

'''
torch.repeat_interleave(
    input,       # 输入张量（这里是 valid_lens）
    repeats,     # 每个元素重复的次数（这里是 shape[1]）
    dim=None,    # 要重复的维度（可选，默认 None）
    output_size=None  # 可选，指定输出尺寸
)
'''
def masked_softmax(X,valid_lens):   #X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X,dim=-1)
    else:
        shape=X.shape
        if valid_lens.dim()==1:
            valid_lens=torch.repeat_interleave(valid_lens,shape[1])
        else:
            valid_lens=valid_lens.reshape(-1)
        X=d2l.sequence_mask(X.reshape(-1,shape[-1]),valid_lens,
                            value=-1e6)
        return nn.functional.softmax(X.reshape(shape),dim=-1)
# X=torch.rand(2,2,4)
# print(X)
# print(masked_softmax(X,torch.tensor([[1,3],[2,4]])))

'''
    假设我们要把中文“我爱编程”翻译成英文 "I love coding"。

    源句子 (Key/Value)：“我 爱 编 程” → 这里 m=4。

    目标句子 (Query)：假设翻译进行了一半，已经生成了 "I love"，现在要预测下一个词 → 这里 n=2。

    此时，注意力机制的作用就是：用这 2 个英文单词（Query），去回过头看那 4 个中文单词（Key），算出 2×4 个评分，决定哪个中文词对下一个英文词最重要。
'''


class AdditiveAttention(nn.Module):
    def __init__(self,key_size,query_size,num_hiddens,dropout,**kwargs):
        super(AdditiveAttention,self).__init__(**kwargs)
        self.W_k=nn.Linear(key_size,num_hiddens,bias=False)
        self.W_q=nn.Linear(query_size,num_hiddens,bias=False)
        self.w_v=nn.Linear(num_hiddens,1,bias=False)
        self.dropout=nn.Dropout(dropout) #随即丢弃一些神经元，提高泛化能力

    def forward(self,queries,keys,values,valid_lens):
        queries,keys=self.W_q(queries),self.W_k(keys) #现在维数相同，都是num_hiddens
        '''
        queries: (batch, n, query_size) → (batch, n, num_hiddens)

        keys: (batch, m, key_size) → (batch, m, num_hiddens)
        '''
        features=queries.unsqueeze(2)+keys.unsqueeze(1)
        features=torch.tanh(features)
        scores=self.w_v(features).squeeze(-1) #(batch_size,query_size,key_size)
        self.attention_weights=masked_softmax(scores,valid_lens) #不影响维度
        return torch.bmm(self.dropout(self.attention_weights),values) #(batch_size,key_size,value_size)

keys=torch.ones((2,10,2))

values=torch.arange(40,dtype=torch.float32).reshape(1,10,4).repeat(2,1,1)
valid_lens=torch.tensor([2,6])
# attention=AdditiveAttention(2,20,8,0.1)
# attention.eval()
# print(attention(queries,keys,values,valid_lens))
#
# show_heatmaps(attention.attention_weights.reshape((1,1,2,10)),
#               xlabel='keys',ylabel='queries')



'''
scaled dot-product attention
'''

class DotProductAttention(nn.Module):
    def __init__(self,dropout,**kwargs):
        super(DotProductAttention,self).__init__(**kwargs)
        self.dropout=nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)

    def forward(self,queries,keys,values,valid_lens=None):

        d=queries.shape[-1]

        scores=torch.bmm(queries,keys.transpose(1,2)/math.sqrt(d))
        self.attentention_weights=masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

queries=torch.normal(0,1,(2,1,2))
attention=DotProductAttention(dropout=0.5)



import math
import torch
from torch import nn
from d2l import torch as d2l
from attention_score_function import DotProductAttention
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
        self.attention=DotProductAttention(dropout)
        self.num_hiddens=num_hiddens
        self.d_k=num_hiddens//num_heads

        #Linear layer
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v=nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        #        return self.W_o(output_concat)
        # 将上面三个进行注意力计算后再拼接再经过一个线性层


    def forward(self,queries,keys,values,valid_lens):
        queries=transpose_qkv(self.W_q(queries), self.num_heads)
        keys=transpose_qkv(self.W_k(keys), self.num_heads)
        values=transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens=torch.repeat_interleave(
                valid_lens,repeats=self.num_heads,dim=0
            )

        output,attention_weights=self.attention(queries,keys,values,valid_lens)

        attention_weights=attention_weights.reshape(-1,self.num_heads,attention_weights.shape[1],attention_weights.shape[2])

        output_concat=transpose_output(output,self.num_heads)
        return self.W_o(output_concat),attention_weights

    def get_head_importance_L2(self):

        head_importance=[]
        for head_idx in range(self.num_heads):
            #num_hiddens = num_heads × d_k（总特征维度 = 头数 × 每个头的特征维度）
            '''
            我们的目标是从 W_q 的参数中，单独提取出 “第 head_idx 个头” 对应的权重，因为每个头负责独立的注意力计算。
            '''
            q_head_weight=self.W_q.weight[:,head_idx*self.d_k:(head_idx+1)*self.d_k]
            '''
            self.W_q.weight 的形状是：(num_hiddens, query_size)
            :（行切片）：表示取所有行（对应 query_size 维度，即原始查询的所有特征）；
            head_idx*self.d_k : (head_idx+1)*self.d_k（列切片）：表示取第 head_idx 个头对应的列范围（对应 num_hiddens 维度中该头的特征区间）。
            '''
            k_head_weight=self.W_k.weight[:,head_idx*self.d_k:(head_idx+1)*self.d_k]

            v_head_weight=self.W_v.weight[:,head_idx*self.d_k:(head_idx+1)*self.d_k]

            o_head_weight=self.W_o.weight[head_idx*self.d_k:(head_idx+1)*self.d_k,:]


            l2_norm=(
                torch.norm(q_head_weight)+
                torch.norm(k_head_weight)+
                torch.norm(v_head_weight)+
                torch.norm(o_head_weight)
            )
            head_importance.append(l2_norm)
        return head_importance


    def get_head_importance_var(self):
        head_importance=[]
        output,attention_weights=self.forward(queries,keys,values,valid_lens)
        for head_idx in range(self.num_heads):
            weights=attention_weights[:,head_idx,:,:]
            head_var_sum=0
            for i in range(weights.shape[0]):
                head_var=torch.var(weights[i])
                head_var_sum+=head_var
            head_importance.append(head_var_sum/attention_weights.shape[0])
        return head_importance




    def prune_least_importance_head(self):
        head_importance=self.get_head_importance_var()

        least_important_idx=torch.argmin(torch.tensor(head_importance)).item()
        print(f"裁剪第 {least_important_idx} 个头（重要性分数：{head_importance[least_important_idx]:.4f}）")

        new_num_heads=self.num_heads-1
        new_num_hiddens=new_num_heads*self.d_k
        new_attention=MultiheadAttention(
            key_size=self.W_k.in_features,
            query_size=self.W_q.in_features,
            value_size=self.W_v.in_features,
            num_heads=new_num_heads,
            num_hiddens=new_num_hiddens,
            dropout=self.attention.dropout.p,
            bias=self.W_q.bias is not None
        )


        with torch.no_grad():
            for layer_name in ['W_q','W_k','W_v']:
                '''
                self：指原始的多头注意力模型实例（裁剪前的模型）；
                layer_name：循环变量，依次取 'W_q'/'W_k'/'W_v'；
                作用：动态获取原始模型中当前要处理的线性层（W_q/W_k/W_v），赋值给 src_layer。
                比如当 layer_name='W_q' 时，这行就等于 src_layer = self.W_q；当 layer_name='W_k' 时，等于 src_layer = self.W_k。
                '''
                src_layer=getattr(self,layer_name)
                dst_layer=getattr(new_attention,layer_name) #处理新的attention模型

                src_weight=src_layer.weight.data   # [num_hiddens, in_dim]
                mask=torch.ones(self.num_heads,dtype=bool)
                mask[least_important_idx]=False

                weight_per_head=src_weight.reshape(-1,self.num_heads,self.d_k)
                weight_kept = weight_per_head[:, mask, :].reshape(-1, new_num_hiddens)  # [in_dim, new_num_hiddens]
                dst_layer.weight.data=weight_kept.T # Linear层权重是 [out_dim, in_dim]，需要转置


                if src_layer.bias is not None:
                    bias_per_head=src_layer.bias.data.reshape(self.num_heads,self.d_k)
                    bias_kept=bias_per_head[mask,:].reshape(-1)
                    dst_layer.bias.data=bias_kept


                src_o_weight=self.W_o.weight.data

                src_o_weight_input = src_o_weight.reshape(self.num_hiddens, self.num_heads, self.d_k)[
                    :, mask, :].reshape(self.num_hiddens, new_num_hiddens)
                # 筛选输出维度（裁剪对应输出通道）
                src_o_weight_output = src_o_weight.reshape(self.num_heads, self.d_k, self.num_hiddens)[
                    mask, :, :].reshape(new_num_hiddens, self.num_hiddens)

                new_o_weight=src_o_weight_input[:new_num_hiddens,:]
                new_attention.W_o.weight.data=new_o_weight

                if self.W_o.bias is not None:
                    bias_per_head = self.W_o.bias.data.reshape(self.num_heads, self.d_k)
                    bias_kept = bias_per_head[mask, :].reshape(-1)
                    new_attention.W_o.bias.data = bias_kept
            return new_attention


# 1. 创建原始多头注意力模型
batch_size = 2
seq_len_q = 10
seq_len_kv = 20
num_heads = 8  # 8个头
num_hiddens = 512
dropout = 0.1

model = MultiheadAttention(
    key_size=512, query_size=512, value_size=512,
    num_hiddens=num_hiddens, num_heads=num_heads, dropout=dropout
)

# 2. 生成测试输入
queries = torch.randn(batch_size, seq_len_q, 512)
keys = torch.randn(batch_size, seq_len_kv, 512)
values = torch.randn(batch_size, seq_len_kv, 512)
valid_lens = torch.tensor([8, 10])  # 有效长度

# 3. 原模型前向输出
original_output = model(queries, keys, values, valid_lens)

# 4. 裁剪最不重要的头
pruned_model = model.prune_least_importance_head()

# 5. 裁剪后模型前向输出
# 5. 裁剪后模型前向输出
pruned_output, pruned_attention_weights = pruned_model(queries, keys, values, valid_lens)
print(f"裁剪后模型输出形状：{pruned_output.shape}")  # (2, 10, 448)（512 - 64 = 448）

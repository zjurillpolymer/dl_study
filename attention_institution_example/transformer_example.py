import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# ================= 完全手写的点积注意力（修复核心：保存attention_weights） =================
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        # 新增：初始化注意力权重属性，供外部访问
        self.attention_weights = None

    # queries: (batch_size * num_heads, query_seq_len, depth)
    # keys: (batch_size * num_heads, key_seq_len, depth)
    # values: (batch_size * num_heads, value_seq_len, depth)
    # valid_lens: (batch_size * num_heads,) 或 (batch_size * num_heads, query_seq_len)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 交换 keys 的最后两维以便矩阵乘法: (batch*heads, depth, seq_len_k)
        keys = keys.transpose(1, 2)

        # 计算分数: (batch*heads, seq_len_q, seq_len_k)
        scores = torch.bmm(queries, keys) / math.sqrt(d)

        if valid_lens is not None:
            # 生成掩码
            mask = self.sequence_mask(scores, valid_lens)
            scores = scores.masked_fill(mask, -1e9)

        # Softmax + Dropout
        attention_weights = nn.functional.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # 核心修复：把注意力权重保存为实例属性
        self.attention_weights = attention_weights

        # 加权求和: (batch*heads, seq_len_q, depth)
        return torch.bmm(attention_weights, values)

    def sequence_mask(self, X, valid_lens):
        # X shape: (batch*heads, seq_len_q, seq_len_k)
        # valid_lens shape: (batch*heads,) or (batch*heads, seq_len_q)
        batch_size_times_heads, seq_len_q, seq_len_k = X.shape

        if valid_lens.dim() == 1:
            # 情况 A: valid_lens 是每个样本的有效长度 (batch*heads,)
            # 需要生成 (batch*heads, 1, seq_len_k) 的掩码
            valid_lens = valid_lens.unsqueeze(1).unsqueeze(2)  # (B*H, 1, 1)
            # 创建序列索引 (1, 1, seq_len_k)
            seq_range = torch.arange(seq_len_k, device=X.device).view(1, 1, -1)
            mask = seq_range >= valid_lens  # (B*H, 1, seq_len_k)
        else:
            # 情况 B: valid_lens 是每个 query 位置的有效长度 (batch*heads, seq_len_q)
            # 需要生成 (batch*heads, seq_len_q, seq_len_k) 的掩码
            valid_lens = valid_lens.unsqueeze(2)  # (B*H, seq_len_q, 1)
            seq_range = torch.arange(seq_len_k, device=X.device).view(1, 1, -1)
            mask = seq_range >= valid_lens  # (B*H, seq_len_q, seq_len_k)

        return mask


# ================= 完全手写的多头注意力 =================
class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)

        # 初始化查询、键、值的线性变换层
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=bias)

        # 输出线性变换层
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries, keys, values 形状: (batch_size, seq_len, num_hiddens)
        batch_size, seq_len, _ = queries.shape

        # 1. 线性变换
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)

        # 2. 多头拆分
        queries = self._transpose_qkv(queries, self.num_heads)
        keys = self._transpose_qkv(keys, self.num_heads)
        values = self._transpose_qkv(values, self.num_heads)

        # 3. 处理有效长度
        if valid_lens is not None:
            # valid_lens 原始形状: (batch_size,) 或 (batch_size, seq_len)
            # 需要重复 num_heads 次，变成 (batch_size * num_heads, ...)
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # 4. 计算注意力输出
        # output 形状: (batch_size * num_heads, seq_len, num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # 5. 多头合并
        output_concat = self._transpose_output(output, self.num_heads)

        # 6. 最终线性变换
        return self.W_o(output_concat)

    def _transpose_qkv(self, X, num_heads):
        """为了多注意力头的并行计算而转换张量形状"""
        batch_size, seq_len, _ = X.shape
        # (batch_size, seq_len, num_heads, num_hiddens / num_heads)
        X = X.reshape(batch_size, seq_len, num_heads, -1)
        # (batch_size, num_heads, seq_len, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3)
        # (batch_size * num_heads, seq_len, num_hiddens / num_heads)
        return X.reshape(-1, seq_len, X.shape[-1])

    def _transpose_output(self, X, num_heads):
        """逆转 transpose_qkv 的操作"""
        batch_size_times_num_heads, seq_len, _ = X.shape
        batch_size = batch_size_times_num_heads // num_heads
        # (batch_size, num_heads, seq_len, num_hiddens / num_heads)
        X = X.reshape(batch_size, num_heads, seq_len, -1)
        # (batch_size, seq_len, num_heads, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3)
        # (batch_size, seq_len, num_hiddens)
        return X.reshape(batch_size, seq_len, -1)

# ====================== 1. 超参数配置 ======================
# 模型超参数
vocab_size_src = 10000  # 源语言词表大小（比如英语）
vocab_size_tgt = 8000   # 目标语言词表大小（比如中文）
key_size = 512
query_size = 512
value_size = 512
num_hiddens = 512       # 词向量维度
norm_shape = [512]      # LayerNorm的归一化维度
ffn_num_input = 512
ffn_num_hiddens = 2048  # FFN隐藏层维度
num_heads = 8           # 多头注意力头数
num_layers = 6          # Encoder/Decoder层数
dropout = 0.1           # dropout概率（0~1之间）

# 训练超参数
batch_size = 32
lr = 0.001
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 2. 数据准备（模拟/真实数据均可） ======================
# 模拟训练数据：替换成你的真实数据（源语言序列，目标语言序列）
# 维度说明：[batch_size, seq_len]，每个元素是词表中的索引
def generate_synthetic_data(batch_size, seq_len_src, seq_len_tgt):
    """生成模拟数据：源语言序列+目标语言序列"""
    src_data = torch.randint(1, vocab_size_src, (batch_size, seq_len_src))  # 1~vocab_size-1（0为padding）
    tgt_data = torch.randint(1, vocab_size_tgt, (batch_size, seq_len_tgt))
    # 生成有效长度（模拟非padding部分）
    src_valid_lens = torch.randint(1, seq_len_src+1, (batch_size,))
    tgt_valid_lens = torch.randint(1, seq_len_tgt+1, (batch_size,))
    return src_data, tgt_data, src_valid_lens, tgt_valid_lens

# 构建DataLoader
seq_len_src = 10  # 源语言序列长度
seq_len_tgt = 12  # 目标语言序列长度
src_data, tgt_data, src_valid_lens, tgt_valid_lens = generate_synthetic_data(
    1000, seq_len_src, seq_len_tgt  # 生成1000条样本
)
dataset = TensorDataset(src_data, tgt_data, src_valid_lens, tgt_valid_lens)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ====================== 位置前馈网络 ======================
class PositionWiseFFN(nn.Module): #维度没有变化
    def __init__(self,ffn_num_inputs,ffn_num_hiddens,ffn_num_outputs,**kwargs):
        super(PositionWiseFFN,self).__init__(**kwargs)
        self.dense1=nn.Linear(ffn_num_inputs,ffn_num_hiddens)
        self.relu=nn.ReLU()
        self.dense2=nn.Linear(ffn_num_hiddens,ffn_num_outputs)

    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X)))

# ====================== 残差连接和层规范化 ======================
class AddNorm(nn.Module):
    def __init__(self,normalized_shape,dropout,**kwargs):
        super(AddNorm,self).__init__(**kwargs)
        self.dropout=nn.Dropout(dropout)
        self.ln=nn.LayerNorm(normalized_shape)

    def forward(self,X,Y):
        return self.ln(X+self.dropout(Y))

# ====================== Encoder Block ======================
class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, dropout, num_heads, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            dropout=dropout,
            bias=use_bias
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens
        )
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

# ====================== Transformer Encoder ======================
class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                    num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                    num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder,self).__init__(**kwargs)
        self.num_hiddens=num_hiddens
        self.embedding=nn.Embedding(vocab_size,num_hiddens) # 词索引→词向量
        self.pos_encoding=d2l.PositionalEncoding(num_hiddens,dropout) #位置编码
        self.blks=nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              dropout, num_heads, use_bias)) # 修正参数顺序

    def forward(self,X,valid_lens,*args):
        X=self.pos_encoding(self.embedding(X))*math.sqrt(self.num_hiddens)
        self.attention_weights=[None]*len(self.blks)
        for i,blk in enumerate(self.blks):
            X=blk(X,valid_lens)
            # 现在可以正确访问注意力权重了
            self.attention_weights[i]=blk.attention.attention.attention_weights
        return X

# ====================== Decoder Block ======================
class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        dropout, i, **kwargs):
        super(DecoderBlock,self).__init__(**kwargs)
        self.i=i
        self.attention1 = MultiHeadAttention(
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            dropout=dropout,
            bias=False
        )
        self.addnorm1=AddNorm(norm_shape,dropout)
        self.attention2 = MultiHeadAttention(
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            dropout=dropout,
            bias=False
        )
        self.addnorm2=AddNorm(norm_shape,dropout)
        self.ffn=PositionWiseFFN(
            ffn_num_input,ffn_num_hiddens,num_hiddens)
        self.addnorm3=AddNorm(norm_shape,dropout)

    def forward(self,X,state):
        enc_outputs,enc_valid_lens=state[0],state[1]
        if state[2][self.i] is None:
            key_values=X
        else:
            key_values=torch.cat((state[2][self.i],X),axis=1)
            state[2][self.i]=key_values

        if self.training:
            batch_size,num_steps,_=X.shape
            dec_valid_lens=torch.arange(
                1,num_steps+1,device=X.device
            ).repeat(batch_size,1)
        else:
            dec_valid_lens=None

        X2=self.attention1(X,key_values,key_values,dec_valid_lens)
        Y=self.addnorm1(X,X2)
        Y2=self.attention2(Y,enc_outputs,enc_outputs,enc_valid_lens)
        Z=self.addnorm2(Y,Y2)

        return self.addnorm3(Z,self.ffn(Z)),state

# ====================== Transformer Decoder ======================
class TransformerDecoder(d2l.Decoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                   num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                   num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder,self).__init__(**kwargs)
        self.num_hiddens=num_hiddens
        self.num_layers=num_layers
        self.embedding=nn.Embedding(vocab_size,num_hiddens) #embedding层
        self.pos_encoding=d2l.PositionalEncoding(num_hiddens,dropout) #位置编码
        self.blks=nn.Sequential()#加入block
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, i)
                                 )
        self.dense=nn.Linear(num_hiddens,vocab_size) # 全连接层

    def init_state(self, enc_outputs,enc_valid_lens, *args):
        return [enc_outputs,enc_valid_lens,[None]*self.num_layers]

    def forward(self,X,state):
        X=self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self._attention_weights=[[None] *len(self.blks) for _ in range(2)]
        for i,blk in enumerate(self.blks):
            X,state=blk(X,state)
            # 解码器也能正确访问注意力权重
            self._attention_weights[0][i]=blk.attention1.attention.attention_weights
            self._attention_weights[1][i]=blk.attention2.attention.attention_weights

        return self.dense(X),state

    def attention_weights(self):
        return self._attention_weights

# ====================== 3. 初始化编码器和解码器 ======================
encoder = TransformerEncoder(
    vocab_size=vocab_size_src,
    key_size=key_size, query_size=query_size, value_size=value_size,
    num_hiddens=num_hiddens, norm_shape=norm_shape,
    ffn_num_input=num_hiddens, ffn_num_hiddens=ffn_num_hiddens,
    num_heads=num_heads, num_layers=num_layers, dropout=dropout
).to(device)

decoder = TransformerDecoder(
    vocab_size=vocab_size_tgt,
    key_size=key_size, query_size=query_size, value_size=value_size,
    num_hiddens=num_hiddens, norm_shape=norm_shape,
    ffn_num_input=num_hiddens, ffn_num_hiddens=ffn_num_hiddens,
    num_heads=num_heads, num_layers=num_layers, dropout=dropout
).to(device)

# ====================== 4. 训练配置 ======================
# 损失函数：交叉熵（忽略padding，padding索引为0）
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
# 优化器：Adam（Transformer常用）
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

# ====================== 5. 核心训练循环 ======================
def train_epoch(encoder, decoder, dataloader, loss_fn, optimizer, device):
    """单轮训练"""
    encoder.train()
    decoder.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (src, tgt, src_valid_lens, tgt_valid_lens) in enumerate(dataloader):
        # 数据移到设备上
        src = src.to(device)
        tgt = tgt.to(device)
        src_valid_lens = src_valid_lens.to(device)

        # 步骤1：编码器编码源语言
        enc_outputs = encoder(src, src_valid_lens)

        # 步骤2：解码器输入处理（训练时用教师强制，输入tgt[:, :-1]，预测tgt[:, 1:]）
        tgt_input = tgt[:, :-1]  # 目标输入：去掉最后一个词（<EOS>）
        tgt_label = tgt[:, 1:]  # 目标标签：去掉第一个词（<BOS>）

        # 步骤3：初始化解码器状态
        dec_state = decoder.init_state(enc_outputs, src_valid_lens)

        # 步骤4：解码器解码
        dec_outputs, _ = decoder(tgt_input, dec_state)

        # 步骤5：计算损失（调整维度：[batch, seq_len, vocab] → [batch*seq_len, vocab]）
        batch_size, seq_len, vocab_size = dec_outputs.shape
        loss = loss_fn(
            dec_outputs.reshape(-1, vocab_size),
            tgt_label.reshape(-1)
        )

        # 步骤6：反向传播+更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累计损失
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # 打印批次信息
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

    # 返回本轮平均损失
    return total_loss / total_samples

# 开始训练
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    avg_loss = train_epoch(encoder, decoder, dataloader, loss_fn, optimizer, device)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

# ====================== 6. 保存模型 ======================
torch.save({
    'encoder_state_dict': encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'avg_loss': avg_loss
}, 'transformer_model.pth')
print("\nModel saved to transformer_model.pth")


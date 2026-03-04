import collections
import math
import torch
from torch import nn
# 假设已安装 d2l: pip install d2l
import d2l
from machine_translation_data_base import load_data_nmt


def try_gpu(i=0):
    """
    如果存在 GPU，就返回 gpu(i)，否则返回 cpu()。

    参数:
        i (int): 想要使用的 GPU 索引，默认为 0。

    返回:
        torch.device: 设备对象 ('cuda:0' 或 'cpu')
    """
    if torch.cuda.is_available():
        # 检查请求的 GPU 索引是否有效
        if i < torch.cuda.device_count():
            return torch.device(f'cuda:{i}')
        else:
            print(f"警告：请求的 GPU 索引 {i} 超出范围 (可用数量: {torch.cuda.device_count()})，回退到 cuda:0")
            return torch.device('cuda:0')
    else:
        # 如果没有可用 GPU，返回 CPU
        return torch.device('cpu')

# --- 1. 辅助函数: Sequence Mask ---
def sequence_mask(X, valid_len, value=0):
    """
    在序列中屏蔽不相关的项。
    X: (batch_size, seq_len) 或 (batch_size, seq_len, dim)
    valid_len: (batch_size,)
    """
    if X.dim() == 2:
        batch_size, seq_len = X.shape
    else:
        batch_size, seq_len, _ = X.shape

    # 生成时间步索引 [0, 1, ..., seq_len-1]
    time_steps = torch.arange(seq_len, device=X.device).reshape(1, -1)
    # valid_len shape: (batch_size, 1)
    mask = time_steps < valid_len.reshape(-1, 1)

    # 扩展 mask 以匹配 X 的维度 (如果是 3D)
    if X.dim() == 3:
        mask = mask.unsqueeze(-1).expand_as(X)

    # 创建输出副本并应用掩码
    X_out = X.clone()
    X_out[~mask] = value
    return X_out


# --- 2. 编码器 (Encoder) ---
class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # GRU: input_size=embed_size, hidden_size=num_hiddens, num_layers
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # X shape: (batch_size, num_steps)
        X = self.embedding(X)  # (batch, steps, embed)
        # RNN 需要 (steps, batch, embed)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        # output: (steps, batch, num_hiddens)
        # state: (num_layers, batch, num_hiddens)
        return output, state


# --- 3. 解码器 (Decoder) ---
class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 输入大小 = 词向量 + 上下文向量(即编码器的隐藏状态)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # 直接返回编码器的最终隐藏状态作为解码器的初始状态
        return enc_outputs[1]

    def forward(self, X, state):
        # X shape: (batch_size, num_steps)
        # 嵌入
        X_emb = self.embedding(X).permute(1, 0, 2)  # (steps, batch, embed)

        # 获取上下文向量 (编码器的最后时刻状态)
        # state shape: (num_layers, batch, num_hiddens)
        # 我们取最后一层的状态作为 context: (1, batch, num_hiddens) -> 重复以匹配时间步
        context = state[-1].unsqueeze(0).repeat(X_emb.shape[0], 1, 1)

        # 拼接输入: (steps, batch, embed + num_hiddens)
        X_and_context = torch.cat((X_emb, context), 2)

        # RNN 前向传播
        output, state = self.rnn(X_and_context, state)

        # 全连接层映射到词表大小
        output = self.dense(output)  # (steps, batch, vocab_size)

        # 转回 (batch, steps, vocab_size) 以便计算 Loss
        output = output.permute(1, 0, 2)
        return output, state


# --- 4. 带掩码的损失函数 ---
class MaskedSoftmaxCELoss(nn.Module):
    def __init__(self):
        super(MaskedSoftmaxCELoss, self).__init__()
        # reduction='none' 以便我们手动应用 mask
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, label, valid_len):
        """
        pred: (batch_size, seq_len, vocab_size)
        label: (batch_size, seq_len)
        valid_len: (batch_size,)
        """
        # 确保维度一致
        weights = torch.ones_like(label, dtype=torch.float32)
        weights = sequence_mask(weights, valid_len, value=0)

        # CrossEntropyLoss 期望输入格式: (N, C) 或 (N, C, d1, ...)
        # 这里我们需要对每个时间步计算 loss，所以交换维度
        # pred permute: (batch, seq, vocab) -> (batch, vocab, seq) ?
        # 不，nn.CrossEntropyLoss 处理多维输入时，C 是第 1 维 (index 1)。
        # 输入应为 (batch, vocab, seq_len) 或者我们将 batch 和 seq 展平。
        # 标准做法：将 (batch, seq, vocab) 转为 (batch * seq, vocab)
        # 标签转为 (batch * seq)

        batch_size, seq_len, vocab_size = pred.shape
        pred = pred.reshape(-1, vocab_size)
        label = label.reshape(-1)
        weights = weights.reshape(-1)

        unweighted_loss = self.loss_fn(pred, label)  # (batch * seq,)

        # 应用权重
        weighted_loss = unweighted_loss * weights

        # 恢复形状并求平均 (仅对有效 token 求平均)
        weighted_loss = weighted_loss.reshape(batch_size, seq_len)

        # 总损失 / 有效 token 总数
        loss = weighted_loss.sum() / weights.sum()
        return loss


# --- 5. 训练函数 ---
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            # GRU 的参数命名可能因版本而异，通常遍历 named_parameters 更安全
            for name, param in m.named_parameters():
                if 'weight' in name and len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)

    net.apply(xavier_init_weights)
    net.to(device)
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()

    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])

    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 累加 loss 总和 和 token 总数

        for batch in data_iter:
            optimizer.zero_grad()
            # 数据通常在 d2l 的 DataLoader 中已经是 tensor 列表
            # X: source, X_valid_len, Y: target, Y_valid_len
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]

            # 构造解码器输入: <bos> + Y[:, :-1]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # (batch, steps)

            # 前向传播
            # d2l.EncoderDecoder 的 forward 通常定义为 forward(enc_X, dec_X)
            # 它会自动调用 encoder 和 decoder
            Y_hat, _ = net(X, dec_input)  # Output: (batch, steps, vocab)

            # 计算损失 (注意 Y_hat 对应的是 dec_input 的每一步预测，目标是 Y)
            # Y_hat[:, :, :] 对应预测，Y 对应真实标签
            # 由于 dec_input 长度和 Y 相同，直接计算
            l = loss(Y_hat, Y, Y_valid_len)

            l.backward()
            d2l.grad_clipping(net, 1)

            num_tokens = Y_valid_len.sum()
            optimizer.step()  # 【修正】加上括号

            with torch.no_grad():
                metric.add(l * num_tokens, num_tokens)

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            animator.add(epoch + 1, (metric[0] / metric[1],))

    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')


# --- 6. 执行示例 (需要 d2l 环境和数据) ---
if __name__ == "__main__":
    # 设置超参数
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, try_gpu()

    print(f"Running on device: {device}")

    # 加载数据 (这会下载 fr-en 数据集，首次运行较慢)
    # 注意：d2l.load_data_nmt 返回的是 train_iter, src_vocab, tgt_vocab
    try:
        train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

        # 初始化模型
        encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
        decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
        net = d2l.EncoderDecoder(encoder, decoder)

        # 开始训练
        train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    except Exception as e:
        print(f"Error during execution (likely due to missing d2l data or environment): {e}")
        print("Please ensure 'd2l' package is installed and internet connection is available for data download.")
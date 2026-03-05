import torch
import torch.nn as nn
import torch.optim as optim
import math
import collections
import random
from typing import List, Tuple


# ==========================================
# 1. 基础组件：注意力机制
# ==========================================
class AdditiveAttention(nn.Module): #加性注意力机制
    def __init__(self, num_hiddens, dropout=0.0):
        super().__init__()
        # W_k: keys 投影, W_q: query 投影, W_v: values 投影 (这里 values 通常不需要投影，直接加权)
        # 加性注意力公式: a = w_v^T * tanh(W_k * k + W_q * q)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, query, key, value, valid_lens=None):
        """
        query: (batch_size, num_queries, num_hiddens)
        key:   (batch_size, num_kv, num_hiddens)
        value: (batch_size, num_kv, num_hiddens)
        valid_lens: (batch_size,) or (batch_size, num_queries)
        """
        num_queries = query.shape[1]
        num_kv = key.shape[1]

        # 扩展 query 以匹配 key 的时间步维度 (用于广播相加)
        # query: (B, num_queries, H) -> (B, num_queries, 1, H)
        # key:   (B, num_kv, H)       -> (B, 1, num_kv, H)
        query_expanded = self.W_q(query).unsqueeze(2)
        key_expanded = self.W_k(key).unsqueeze(1)

        # 相加后通过 tanh: (B, num_queries, num_kv, H)
        features = torch.tanh(query_expanded + key_expanded)

        # 投影到标量分数: (B, num_queries, num_kv, 1) -> (B, num_queries, num_kv)
        scores = self.w_v(features).squeeze(-1)

        # 应用掩码
        if valid_lens is not None:
            # valid_lens shape: (batch_size,)
            # 需要生成 mask: (batch_size, 1, num_kv) 以便广播到 (B, num_queries, num_kv)
            mask = self._sequence_mask(valid_lens, num_kv)
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)

        self.attention_weights = torch.softmax(scores, dim=-1)

        # 加权求和: (B, num_queries, num_kv) @ (B, num_kv, H) -> (B, num_queries, H)
        context = torch.bmm(self.dropout(self.attention_weights), value)
        return context

    def _sequence_mask(self, X, maxlen=None, value=0):
        # 生成掩码: True 表示需要掩盖的位置
        if maxlen is None:
            maxlen = X.max()
        # X: (batch,), 范围 [0, maxlen]
        # 生成 [0, 1, ..., maxlen-1]
        grid = torch.arange(maxlen, dtype=torch.float32, device=X.device).reshape(1, -1)
        # mask: (batch, maxlen), 如果 index >= len 则为 True
        mask = grid >= X.unsqueeze(1).float()
        return mask


# ==========================================
# 2. 编码器 (Encoder)
# ==========================================
class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers

    def forward(self, X, valid_lens):
        # X: (batch, seq_len)
        embeddings = self.embedding(X)  # (batch, seq_len, embed)
        # GRU 需要 (seq_len, batch, embed)
        embeddings = embeddings.permute(1, 0, 2)

        outputs, hidden_state = self.rnn(embeddings)
        # outputs: (seq_len, batch, num_hiddens)
        # hidden_state: (num_layers, batch, num_hiddens)
        return outputs, hidden_state


# ==========================================
# 3. 解码器 (Decoder with Attention)
# ==========================================
class Seq2SeqAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0):
        super().__init__()
        self.attention = AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 输入维度 = embed_size + context_vector (num_hiddens)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self._attention_weights = []

    def init_state(self, enc_outputs, enc_valid_lens):
        outputs, hidden_state = enc_outputs
        # 转置 outputs 为 (batch, seq_len, hidden) 方便 Attention 计算
        return {
            'enc_outputs': outputs.permute(1, 0, 2),
            'hidden_state': hidden_state,
            'enc_valid_lens': enc_valid_lens
        }

    def forward(self, X, state):
        # X: (batch, seq_len) 目标序列输入 (Teacher Forcing)
        enc_outputs = state['enc_outputs']  # (batch, src_len, hidden)
        hidden_state = state['hidden_state']  # (layers, batch, hidden)
        enc_valid_lens = state['enc_valid_lens']

        X_emb = self.embedding(X)  # (batch, seq_len, embed)
        # 转换为 (seq_len, batch, embed) 用于循环
        X_emb = X_emb.permute(1, 0, 2)

        outputs = []
        self._attention_weights = []

        for x_t in X_emb:
            # x_t: (batch, embed)
            # Query: 使用最后一层隐藏状态 (batch, hidden) -> (batch, 1, hidden)
            query = hidden_state[-1:].permute(1, 0, 2)  # 取最后一层，形状 (1, batch, hidden) -> permute -> (batch, 1, hidden)
            # 或者更直接:
            query = hidden_state[-1].unsqueeze(1)  # (batch, 1, hidden)

            # Context: (batch, 1, hidden)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)

            # 拼接: (batch, 1, hidden + embed)
            x_concat = torch.cat((context, x_t.unsqueeze(1)), dim=-1)

            # RNN 输入需要 (seq_len=1, batch, input_size)
            out, hidden_state = self.rnn(x_concat.permute(1, 0, 2), hidden_state)
            # out: (1, batch, hidden)

            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)

        # 拼接所有时间步: (seq_len, batch, hidden)
        outputs = torch.cat(outputs, dim=0)
        # 全连接映射到词表: (seq_len, batch, vocab)
        outputs = self.dense(outputs)
        # 转置回 (batch, seq_len, vocab)
        outputs = outputs.permute(1, 0, 2)

        # 更新状态供下一步使用（虽然训练时是一次性传入整个序列，但为了接口统一）
        state['hidden_state'] = hidden_state
        return outputs, state

    def get_attention_weights(self):
        return self._attention_weights


# ==========================================
# 4. 封装模型 & 损失函数
# ==========================================
class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, tgt_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_pad_idx = tgt_pad_idx

    def forward(self, src, tgt, src_valid_lens):
        # src: (batch, src_len)
        # tgt: (batch, tgt_len) -> 输入给 Decoder 的是 tgt[:, :-1]
        enc_outputs = self.encoder(src, src_valid_lens)
        dec_state = self.decoder.init_state(enc_outputs, src_valid_lens)

        # Teacher Forcing: 输入是去掉最后一个 token 的目标序列
        dec_input = tgt[:, :-1]
        predictions, _ = self.decoder(dec_input, dec_state)

        # predictions: (batch, tgt_len-1, vocab)
        # target labels: 去掉第一个 token (对应 <bos>) 的目标序列
        labels = tgt[:, 1:]

        return predictions, labels


class MaskedSoftmaxCELoss(nn.Module):
    def __init__(self, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, label, valid_lens):
        # pred: (batch, seq_len, vocab)
        # label: (batch, seq_len)
        # valid_lens: (batch,) 表示每个样本的有效长度

        batch_size, seq_len, vocab_size = pred.shape

        # 计算未掩码的损失
        loss = self.criterion(pred.reshape(-1, vocab_size), label.reshape(-1))
        loss = loss.reshape(batch_size, seq_len)

        # ========== 修复1：将 valid_lens 移到 pred 所在设备 ==========
        valid_lens = valid_lens.to(pred.device)

        # 创建掩码
        # valid_lens: (batch,) -> 需要生成 (batch, seq_len) 的 mask
        device = pred.device  # 获取输入数据所在的设备 (cpu 或 cuda)
        mask = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = mask >= valid_lens.unsqueeze(1)

        # 掩盖掉 padding 部分的 loss
        loss = loss.masked_fill(mask, 0.0)

        # 计算平均损失 (除以有效 token 总数)
        total_valid_tokens = valid_lens.sum().item()
        if total_valid_tokens == 0:
            return 0.0
        return loss.sum() / total_valid_tokens


# ==========================================
# 5. 数据处理 (模拟 NMT 数据)
# ==========================================
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None: tokens = []
        if reserved_tokens is None: reserved_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        counter = collections.Counter(tokens)
        self.idx_to_token = list(reserved_tokens)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in counter.items():
            if freq >= min_freq and token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
        self.pad = 0
        self.bos = 1
        self.eos = 2
        self.unk = 3

    def __len__(self):
        return len(self.idx_to_token)

    def to_indices(self, tokens):
        return [self.token_to_idx.get(t, self.unk) for t in tokens]

    def to_tokens(self, indices):
        return [self.idx_to_token[i] for i in indices]


def build_data(batch_size, num_steps=10, num_examples=1000):
    # 生成一些简单的模拟数据：源序列是随机数字，目标序列是源序列的反转 + 标记
    # 实际使用时可替换为读取文件
    sources = []
    targets = []

    for _ in range(num_examples):
        length = random.randint(5, num_steps - 2)
        src_tokens = [str(random.randint(0, 9)) for _ in range(length)]
        tgt_tokens = ['<bos>'] + src_tokens[::-1] + ['<eos>']

        # 填充
        while len(src_tokens) < num_steps: src_tokens.append('<pad>')
        while len(tgt_tokens) < num_steps + 1: tgt_tokens.append('<pad>')  # tgt 多一个 bos

        sources.append(' '.join(src_tokens[:num_steps]))
        targets.append(' '.join(tgt_tokens[:num_steps + 1]))  # 注意长度

    # 构建词表
    all_tokens = ' '.join(sources + targets).split()
    src_vocab = Vocab(all_tokens, reserved_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])
    tgt_vocab = Vocab(all_tokens, reserved_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])

    def preprocess(line, vocab, max_len):
        tokens = line.split()
        # 截断
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        indices = vocab.to_indices(tokens)
        # 确保长度一致 (这里简单处理，实际可能需要动态 padding)
        return indices

    dataset = []
    for src, tgt in zip(sources, targets):
        src_ids = preprocess(src, src_vocab, num_steps)
        tgt_ids = preprocess(tgt, tgt_vocab, num_steps + 1)  # ========== 修复2：tgt 长度应为 num_steps+1 ==========

        # 简单的 Padding 逻辑确保长度固定
        if len(src_ids) < num_steps: src_ids += [src_vocab.pad] * (num_steps - len(src_ids))
        if len(tgt_ids) < num_steps + 1: tgt_ids += [tgt_vocab.pad] * (num_steps + 1 - len(tgt_ids))  # tgt 长度+1

        # 计算有效长度 (非 pad 的数量)
        src_len = sum(1 for i in src_ids if i != src_vocab.pad)
        tgt_len = sum(1 for i in tgt_ids if i != tgt_vocab.pad)

        dataset.append((src_ids, tgt_ids, src_len, tgt_len))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, src_vocab.pad, tgt_vocab.pad)
    )
    return data_loader, src_vocab, tgt_vocab


def collate_fn(batch, src_pad, tgt_pad):
    srcs, tgts, src_lens, tgt_lens = zip(*batch)
    src_tensor = torch.tensor(srcs, dtype=torch.long)
    tgt_tensor = torch.tensor(tgts, dtype=torch.long)
    # ========== 修复3：有效长度应为 int 类型 (float 会导致后续比较错误) ==========
    src_len_tensor = torch.tensor(src_lens, dtype=torch.int32)
    tgt_len_tensor = torch.tensor(tgt_lens, dtype=torch.int32)
    return src_tensor, tgt_tensor, src_len_tensor, tgt_len_tensor


# ==========================================
# 6. 训练函数
# ==========================================
def train_seq2seq(model, data_iter, lr, num_epochs, tgt_vocab, device):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = MaskedSoftmaxCELoss(tgt_vocab.pad)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for src, tgt, src_valid_len, tgt_valid_len in data_iter:
            src = src.to(device)
            tgt = tgt.to(device)
            src_valid_len = src_valid_len.to(device)
            tgt_valid_len = tgt_valid_len.to(device)  # ========== 修复4：tgt_valid_len 也移到 GPU ==========

            # tgt_valid_len 在 loss 计算中需要减去 1，因为预测长度比输入少 1
            # 这里的 tgt_valid_len 是原始长度，作为 label 的有效长度应该是 max(0, len-1)
            label_valid_len = torch.clamp(tgt_valid_len - 1, min=0)

            optimizer.zero_grad()

            predictions, labels = model(src, tgt, src_valid_len)
            # predictions: (B, T-1, V), labels: (B, T-1)

            loss = loss_fn(predictions, labels, label_valid_len)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / num_batches:.4f}")


# ==========================================
# 7. 预测与 BLEU
# ==========================================
def predict_seq2seq(model, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    model.eval()
    src_tokens = src_sentence.split()
    src_indices = src_vocab.to_indices(src_tokens)
    if len(src_indices) > num_steps:
        src_indices = src_indices[:num_steps]
    else:
        src_indices += [src_vocab.pad] * (num_steps - len(src_indices))

    src_tensor = torch.tensor([src_indices], dtype=torch.long, device=device)
    src_valid_len = torch.tensor([sum(1 for i in src_indices if i != src_vocab.pad)], dtype=torch.int32,
                                 device=device)

    # 编码
    enc_outputs = model.encoder(src_tensor, src_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, src_valid_len)

    # 解码 (贪婪搜索)
    outputs = [tgt_vocab.bos]
    attention_weights_seq = []

    with torch.no_grad():  # ========== 修复5：预测时禁用梯度计算 ==========
        for _ in range(num_steps):
            dec_input = torch.tensor([outputs], dtype=torch.long, device=device)
            pred, dec_state = model.decoder(dec_input, dec_state)
            # pred: (1, 1, vocab)
            next_token = pred.argmax(dim=-1)[0, -1].item()

            outputs.append(next_token)
            attention_weights_seq.append(model.decoder.get_attention_weights()[-1])

            if next_token == tgt_vocab.eos:
                break

    pred_tokens = tgt_vocab.to_tokens(outputs[1:])  # 去掉 bos
    # 清理 eos 和 pad
    try:
        eos_idx = pred_tokens.index('<eos>')
        pred_tokens = pred_tokens[:eos_idx]
    except ValueError:
        pass

    return ' '.join(pred_tokens), attention_weights_seq


def bleu(pred_seq, label_seq, k):
    pred_tokens = pred_seq.split()
    label_tokens = label_seq.split()
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred)) if len_pred > 0 else 0

    for n in range(1, k + 1):
        num_matches, matches = 0, 0
        for i in range(len_pred - n + 1):
            pred_ngram = ' '.join(pred_tokens[i:i + n])
            if pred_ngram in label_seq:  # 简单检查，严谨实现需统计计数
                # 这里简化处理，实际应统计 n-gram 出现次数的最小值
                if pred_ngram in [' '.join(label_tokens[j:j + n]) for j in range(len_label - n + 1)]:
                    num_matches += 1
            matches += 1  # 这里的逻辑需要修正为标准的 BLEU 计算，这里仅作演示框架
        # 简化版 BLEU 计算逻辑 (仅示意，建议使用 nltk.translate.bleu_score)
        if matches > 0:
            score *= math.pow(num_matches / matches, math.pow(0.5, n))

    # 使用 nltk 计算标准 BLEU (如果可用)
    try:
        from nltk.translate import bleu_score
        candidate = pred_tokens
        reference = [label_tokens]
        weights = tuple(1 / k for n in range(1, k + 1))
        return bleu_score.sentence_bleu(reference, candidate, weights=weights)
    except ImportError:
        return score  # 返回简化版分数


# ==========================================
# 8. 主程序执行
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 超参数
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs = 0.005, 50  # 减少 epoch 数以便快速测试

    # 加载数据
    train_iter, src_vocab, tgt_vocab = build_data(batch_size, num_steps, num_examples=2000)

    # 初始化模型
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = Seq2SeqModel(encoder, decoder, tgt_vocab.pad)

    # 训练
    print("Start training...")
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 测试
    print("\nInference Examples:")
    test_cases = [
        ("0 1 2 3 4", "4 3 2 1 0"),  # 模拟反转任务
        ("9 8 7", "7 8 9"),
        ("1 2 3 4 5 6", "6 5 4 3 2 1")
    ]

    for src, expected in test_cases:
        # 构造符合格式的输入 (可能需要补全 pad，但 predict 函数内部处理了)
        translation, _ = predict_seq2seq(net, src, src_vocab, tgt_vocab, num_steps, device)
        # 计算 BLEU 分数
        bleu_score = bleu(translation, expected, k=2)
        print(f"Source: {src} => Predicted: {translation} (Expected: {expected}, BLEU: {bleu_score:.4f})")

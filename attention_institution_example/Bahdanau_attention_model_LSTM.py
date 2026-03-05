import torch
import torch.nn as nn
import torch.optim as optim
import math
import collections
import random
from typing import List, Tuple


# ==========================================
# 1. 基础组件：注意力机制（无修改）
# ==========================================
class AdditiveAttention(nn.Module):
    def __init__(self, num_hiddens, dropout=0.0):
        super().__init__()
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, query, key, value, valid_lens=None):
        num_queries = query.shape[1]
        num_kv = key.shape[1]

        query_expanded = self.W_q(query).unsqueeze(2)
        key_expanded = self.W_k(key).unsqueeze(1)
        features = torch.tanh(query_expanded + key_expanded)
        scores = self.w_v(features).squeeze(-1)

        if valid_lens is not None:
            mask = self._sequence_mask(valid_lens, num_kv)
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)

        self.attention_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(self.dropout(self.attention_weights), value)
        return context

    def _sequence_mask(self, X, maxlen=None, value=0):
        if maxlen is None:
            maxlen = X.max()
        grid = torch.arange(maxlen, dtype=torch.float32, device=X.device).reshape(1, -1)
        mask = grid >= X.unsqueeze(1).float()
        return mask


# ==========================================
# 2. 编码器 (Encoder) - 改为 LSTM
# ==========================================
class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # ========== 修改1：GRU 改为 LSTM ==========
        self.rnn = nn.LSTM(embed_size, num_hiddens, num_layers, dropout=dropout)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers

    def forward(self, X, valid_lens):
        # X: (batch, seq_len)
        embeddings = self.embedding(X)  # (batch, seq_len, embed)
        embeddings = embeddings.permute(1, 0, 2)  # (seq_len, batch, embed)

        # ========== 修改2：接收 LSTM 的输出 (outputs, (hidden, cell)) ==========
        outputs, (hidden_state, cell_state) = self.rnn(embeddings)
        # outputs: (seq_len, batch, num_hiddens)
        # hidden_state: (num_layers, batch, num_hiddens)
        # cell_state: (num_layers, batch, num_hiddens)
        return outputs, (hidden_state, cell_state)  # 返回元组


# ==========================================
# 3. 解码器 (Decoder with Attention) - 适配 LSTM
# ==========================================
class Seq2SeqAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0):
        super().__init__()
        self.attention = AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # ========== 修改3：GRU 改为 LSTM ==========
        self.rnn = nn.LSTM(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self._attention_weights = []

    def init_state(self, enc_outputs, enc_valid_lens):
        outputs, (hidden_state, cell_state) = enc_outputs  # 接收 LSTM 的状态
        return {
            'enc_outputs': outputs.permute(1, 0, 2),
            'hidden_state': hidden_state,  # LSTM 隐藏状态
            'cell_state': cell_state,  # LSTM 细胞状态
            'enc_valid_lens': enc_valid_lens
        }

    def forward(self, X, state):
        enc_outputs = state['enc_outputs']
        hidden_state = state['hidden_state']
        cell_state = state['cell_state']  # 新增：获取细胞状态
        enc_valid_lens = state['enc_valid_lens']

        X_emb = self.embedding(X)
        X_emb = X_emb.permute(1, 0, 2)

        outputs = []
        self._attention_weights = []

        for x_t in X_emb:
            # Query: 使用最后一层隐藏状态
            query = hidden_state[-1].unsqueeze(1)
            # 计算注意力上下文
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # 拼接输入和上下文
            x_concat = torch.cat((context, x_t.unsqueeze(1)), dim=-1)

            # ========== 修改4：LSTM 输入需要 (seq_len, batch, input_size)，并传入 (hidden, cell) ==========
            out, (hidden_state, cell_state) = self.rnn(x_concat.permute(1, 0, 2), (hidden_state, cell_state))

            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)

        # 拼接所有时间步并映射到词表
        outputs = torch.cat(outputs, dim=0)
        outputs = self.dense(outputs)
        outputs = outputs.permute(1, 0, 2)

        # ========== 修改5：更新状态时同时保存 hidden 和 cell ==========
        state['hidden_state'] = hidden_state
        state['cell_state'] = cell_state
        return outputs, state

    def get_attention_weights(self):
        return self._attention_weights


# ==========================================
# 4. 封装模型 & 损失函数（无修改）
# ==========================================
class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, tgt_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_pad_idx = tgt_pad_idx

    def forward(self, src, tgt, src_valid_lens):
        enc_outputs = self.encoder(src, src_valid_lens)
        dec_state = self.decoder.init_state(enc_outputs, src_valid_lens)
        dec_input = tgt[:, :-1]
        predictions, _ = self.decoder(dec_input, dec_state)
        labels = tgt[:, 1:]
        return predictions, labels


class MaskedSoftmaxCELoss(nn.Module):
    def __init__(self, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, label, valid_lens):
        batch_size, seq_len, vocab_size = pred.shape
        loss = self.criterion(pred.reshape(-1, vocab_size), label.reshape(-1))
        loss = loss.reshape(batch_size, seq_len)

        valid_lens = valid_lens.to(pred.device)
        device = pred.device
        mask = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = mask >= valid_lens.unsqueeze(1)

        loss = loss.masked_fill(mask, 0.0)
        total_valid_tokens = valid_lens.sum().item()
        if total_valid_tokens == 0:
            return 0.0
        return loss.sum() / total_valid_tokens


# ==========================================
# 5. 数据处理（无修改）
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
    sources = []
    targets = []

    for _ in range(num_examples):
        length = random.randint(3, num_steps - 2)
        src_tokens = [str(random.randint(0, 9)) for _ in range(length)]
        tgt_tokens = ['<bos>'] + src_tokens[::-1] + ['<eos>']

        src_tokens_padded = src_tokens + ['<pad>'] * (num_steps - len(src_tokens))
        tgt_tokens_padded = tgt_tokens + ['<pad>'] * (num_steps + 1 - len(tgt_tokens))

        sources.append(' '.join(src_tokens_padded))
        targets.append(' '.join(tgt_tokens_padded))

    all_tokens = ' '.join(sources + targets).split()
    src_vocab = Vocab(all_tokens, reserved_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])
    tgt_vocab = Vocab(all_tokens, reserved_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])

    def preprocess(line, vocab, max_len):
        tokens = line.split()
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        indices = vocab.to_indices(tokens)
        return indices

    dataset = []
    for src, tgt in zip(sources, targets):
        src_ids = preprocess(src, src_vocab, num_steps)
        tgt_ids = preprocess(tgt, tgt_vocab, num_steps + 1)

        if len(src_ids) < num_steps: src_ids += [src_vocab.pad] * (num_steps - len(src_ids))
        if len(tgt_ids) < num_steps + 1: tgt_ids += [tgt_vocab.pad] * (num_steps + 1 - len(tgt_ids))

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
    src_len_tensor = torch.tensor(src_lens, dtype=torch.int32)
    tgt_len_tensor = torch.tensor(tgt_lens, dtype=torch.int32)
    return src_tensor, tgt_tensor, src_len_tensor, tgt_len_tensor


# ==========================================
# 6. 训练函数（无修改）
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
            tgt_valid_len = tgt_valid_len.to(device)
            label_valid_len = torch.clamp(tgt_valid_len - 1, min=0)

            optimizer.zero_grad()
            predictions, labels = model(src, tgt, src_valid_len)
            loss = loss_fn(predictions, labels, label_valid_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / num_batches:.4f}")


# ==========================================
# 7. 预测函数（适配 LSTM 状态）
# ==========================================
def predict_seq2seq(model, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    model.eval()
    src_tokens = src_sentence.split()
    src_len = len(src_tokens)
    src_indices = src_vocab.to_indices(src_tokens)

    if len(src_indices) > num_steps:
        src_indices = src_indices[:num_steps]
    else:
        src_indices += [src_vocab.pad] * (num_steps - len(src_indices))

    src_tensor = torch.tensor([src_indices], dtype=torch.long, device=device)
    src_valid_len = torch.tensor([src_len], dtype=torch.int32, device=device)

    # 编码
    enc_outputs = model.encoder(src_tensor, src_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, src_valid_len)

    # 解码
    outputs = [tgt_vocab.bos]
    attention_weights_seq = []

    with torch.no_grad():
        for _ in range(src_len + 2):
            dec_input = torch.tensor([[outputs[-1]]], dtype=torch.long, device=device)
            pred, dec_state = model.decoder(dec_input, dec_state)
            next_token = pred.argmax(dim=-1)[0, 0].item()

            outputs.append(next_token)
            attention_weights_seq.append(model.decoder.get_attention_weights()[-1])

            if next_token == tgt_vocab.eos or len(outputs) >= src_len + 2:
                break

    pred_tokens = tgt_vocab.to_tokens(outputs[1:])
    try:
        eos_idx = pred_tokens.index('<eos>')
        pred_tokens = pred_tokens[:eos_idx]
    except ValueError:
        pass
    pred_tokens = pred_tokens[:src_len]

    return ' '.join(pred_tokens), attention_weights_seq


# ==========================================
# 8. BLEU 计算（无修改）
# ==========================================
def bleu(pred_seq, label_seq, k):
    pred_tokens = pred_seq.split()
    label_tokens = label_seq.split()
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred)) if len_pred > 0 else 0

    for n in range(1, k + 1):
        num_matches, matches = 0, 0
        for i in range(len_pred - n + 1):
            pred_ngram = ' '.join(pred_tokens[i:i + n])
            if pred_ngram in label_seq:
                if pred_ngram in [' '.join(label_tokens[j:j + n]) for j in range(len_label - n + 1)]:
                    num_matches += 1
            matches += 1
        if matches > 0:
            score *= math.pow(num_matches / matches, math.pow(0.5, n))

    try:
        from nltk.translate import bleu_score
        candidate = pred_tokens
        reference = [label_tokens]
        weights = tuple(1 / k for n in range(1, k + 1))
        return bleu_score.sentence_bleu(reference, candidate, weights=weights)
    except ImportError:
        return score


# ==========================================
# 9. 主程序执行
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 优化后的超参数
    embed_size, num_hiddens, num_layers, dropout = 64, 64, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs = 0.005, 100  # 增加训练轮数

    # 加载数据
    train_iter, src_vocab, tgt_vocab = build_data(batch_size, num_steps, num_examples=2000)

    # 初始化 LSTM 模型
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = Seq2SeqModel(encoder, decoder, tgt_vocab.pad)

    # 训练
    print("Start training (LSTM version)...")
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 测试
    print("\nInference Examples (LSTM version):")
    test_cases = [
        ("0 1 2 3 4", "4 3 2 1 0"),
        ("9 8 7", "7 8 9"),
        ("1 2 3 4 5 6", "6 5 4 3 2 1")
    ]

    for src, expected in test_cases:
        translation, _ = predict_seq2seq(net, src, src_vocab, tgt_vocab, num_steps, device)
        bleu_score = bleu(translation, expected, k=2)
        print(f"Source: {src} => Predicted: {translation} (Expected: {expected}, BLEU: {bleu_score:.4f})")

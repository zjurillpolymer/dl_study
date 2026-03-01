import torch
import random

'''
corpus = [
    2, 7, 4, 1, 2, 8, 12, 4, 1, 12, 0, 7, 8, 13, 4, 1, 11, 14, 5, 1, 2, 4, 15, 5, 1, ...
    # ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
    # t  h  e     t  i  m  e     m  a  c  h  i  n  e     b  y     H  G     W  e  l  l  s
]
'''


def random_data_loader(corpus, batch_size, num_steps, shuffle=True):
    corpus = corpus[:batch_size * (len(corpus) // batch_size)]  ### 去除掉末尾的剩余的部分
    n_batches = len(corpus) // batch_size
    data = torch.tensor(corpus, dtype=torch.long).reshape(batch_size, n_batches)

    # 计算可用的起始位置数量
    n_samples = n_batches - num_steps

    if shuffle:
        # 随机打乱所有可能的起始位置
        indices = list(range(n_samples))
        random.shuffle(indices)
        for j in indices:
            X = data[:, j:j + num_steps]
            Y = data[:, j + 1:j + num_steps + 1]
            yield X, Y
    else:
        # 顺序遍历，步长为num_steps避免重叠
        for j in range(0, n_samples, num_steps):
            X = data[:, j:j + num_steps]
            Y = data[:, j + 1:j + num_steps + 1]
            yield X, Y
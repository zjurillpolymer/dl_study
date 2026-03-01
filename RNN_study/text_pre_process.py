import collections
import re
from d2l import torch as d2l
import numpy as np
import torch
import random
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt','090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
        '''
        将所有非字母字符替换为空格（包括标点、数字、特殊符号等）
        '''
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()  ### lines是一维列表，每一个元素都是str，例如'the time machine by h g wells', '', '', ''
# # print(lines)
# for i in range(10):
#     print(lines[i])
# print(f'#文本总行数:{len(lines)}')
# print(type(lines[0]))


'''
逐行将文本行拆分为字符或者词元
'''
def tokenize(lines,token='word'):

    if token=='word':
        return [line.split() for line in lines]
    elif token=='char':
        return [list(line) for line in lines]
    else:
        print("error")

tokens=tokenize(lines) ### tokens是一个二维列表
# for i in range(10):
#     print(tokens[i])
'''
构建词表
用来将词元映射为从0开始的数字索引中
我们将训练集中的所有文档合并在一起，对它们唯一词元进行统计，得到的统计结果称之为语料，然后据其出现频率，给予其一个数字索引
'''

def count_corpus(tokens):
    if len(tokens)==0 or isinstance(tokens[0],list):
        tokens=[token for line in tokens for token in line]
    return collections.Counter(tokens)
# print(count_corpus(tokens).most_common(3)) [('the', 2261), ('i', 1267), ('and', 1245)]


'''
self._token_freqs是一个按frequency从高到低排序的list，元素均为tuple  List[Tuple[str, int]]
self.idx_to_token将idx转为token，是一个list   List[str]
self.token_to_idx将token转为idx，是一个字典    Dict[str, int]
'''
class Vocab:
    def __init__(self, tokens=None,min_freq=0,reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        counter=count_corpus(tokens)

        '''
        counter本身是按插入顺序排序的
        现在按频率排序
        '''
        self._token_freqs=sorted(counter.items(),key=lambda x:x[1],reverse=True)

        '''

        | 索引→token | `_idx_to_token` | 解码：将模型输出转回文本 | `idx=2` → `'the'` |
        | token→索引 | `_token_to_idx` | 编码：将文本转为模型输入 | `'the'` → `idx=2` |
        
        '''

        self.idx_to_token=['<unk>']+reserved_tokens
        self.token_to_idx={token:idx
                           for idx,token in enumerate(self.idx_to_token)} ### 字典： (token:idx)
        for token,freq in self._token_freqs:
            if freq<min_freq:
                break
            if token not in  self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token]=len(self.idx_to_token)-1
                '''token_to_idx前面是特殊词，后面是频率由高到低的一般词，idx即其索引'''

    def __len__(self):
        return len(self.idx_to_token)

    '''
    用法1:单个词 → 单个索引
    用法2：词列表 → 索引列表
    '''
    def __getitem__(self,tokens): ###这里的tokens是输入，可以是单个字符，也可以是元组，也可以是list
        if not isinstance(tokens,(list,tuple)):
            return self.token_to_idx.get(tokens,self.unk)  ### 未知词汇返回unk的索引
        return [self.__getitem__(token) for token in tokens]

    def to_token(self,indices):  ###从token获得idx
        if not isinstance(indices,(list,tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def unk(self):
        return 0

    def token_freqs(self):
        return self._token_freqs  ###token-freq的sorted 字典



vocab=Vocab(tokens)
# print(list(vocab.token_to_idx.items())[:10])
#
#
# for i in range(10):
#     print('token:',tokens[i])
#     print('idx:',vocab[tokens[i]]) ### __getitem__的调用  vocab.__getitem__(tokens[i])  # ✅ 完整写法（但没人这么写）



'''整合所有功能'''
'''
corpus = [
    2, 7, 4, 1, 2, 8, 12, 4, 1, 12, 0, 7, 8, 13, 4, 1, 11, 14, 5, 1, 2, 4, 15, 5, 1, ...
    # ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
    # t  h  e     t  i  m  e     m  a  c  h  i  n  e     b  y     H  G     W  e  l  l  s
]
'''

def load_corpus_time_machine(max_token=-1):
    lines=read_time_machine()
    tokens=tokenize(lines,'char')  ###tokens是一个二维列表
    vocab=Vocab(tokens)

    corpus=[vocab[token] for line in tokens for token in line]
    if max_token>0:
        corpus=corpus[:max_token]
    return corpus,vocab

# corpus,vocab=load_corpus_time_machine()
# token_freqs=vocab._token_freqs
# # print(type(token_freqs))
# # print(token_freqs[:10])
# # ranks = np.arange(1, len(token_freqs) + 1)
# # print(ranks[:10])
# # freqs = [freq for token, freq in token_freqs]
# # print(freqs[:10])
# print(type(corpus))
# print(corpus[:10])


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成小批量序列数据"""
    corpus = corpus[:batch_size * num_steps]
    # 划分为 batch_size 行
    Xs = torch.tensor(corpus[:batch_size * num_steps].reshape(batch_size, -1))
    Ys = torch.tensor(corpus[1:batch_size * num_steps + 1].reshape(batch_size, -1))

    num_batches = (Xs.shape[1] - 1) // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """
    使用顺序分区生成小批量序列数据

    参数:
        corpus: 语料库，词元索引组成的列表
        batch_size: 批量大小
        num_steps: 每个样本的时间步数（序列长度）

    返回:
        生成器，每次yield一个批次的 (X, Y)
    """
    # 1. 计算起始位置偏移量
    # 确保 corpus 能被 batch_size 整除，从 offset 开始截取
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])

    # 2. 将数据 reshapes 成 (batch_size, -1) 的形状
    # 每一行是一个独立的序列
    Xs = Xs.reshape(batch_size, -1)
    Ys = Ys.reshape(batch_size, -1)

    # 3. 计算可以生成多少个完整批次
    num_batches = (Xs.shape[1] - 1) // num_steps

    # 4. 按顺序遍历生成批次
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y



def load_data_time_machine(batch_size, num_steps, use_random_iter=True, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    corpus, vocab = load_corpus_time_machine(max_tokens)

    if use_random_iter:
        data_iter = seq_data_iter_random
    else:
        data_iter = seq_data_iter_sequential

    data_iter_fn = lambda: data_iter(corpus, batch_size, num_steps)

    return data_iter_fn, vocab
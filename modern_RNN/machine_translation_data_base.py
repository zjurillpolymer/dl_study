import os
import torch
from d2l import torch as d2l
from RNN_study.text_pre_process import Vocab


#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
# print(raw_text[:75])


#@save
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
# print(text[:80])
# for i,line in enumerate(text.split('\n')):
#     print(i,line)
#     break   # 0 go .	va !



#@save
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
# print(len(source), len(target))
# print(source[:10])
# print(target[:10])


src_vocab = Vocab(source,min_freq=2,reserved_tokens=['<pad>', '<bos>', '<eos>'])

# print(list(src_vocab.token_to_idx.items())[:48])
# [('<unk>', 0), ('<pad>', 1), ('<bos>', 2), ('<eos>', 3), ('.', 4), ('i', 5), ('you', 6), ('to', 7), ('the', 8), ('?', 9)]
# print(src_vocab[source[0]])
#[47,4]
#对应词表中的go .

'''
如果文本序列长度长于num_steps，那么直接截断
反之，用<pad>词元补齐
'''

def truncate_pad(line,num_steps,padding_token):
    if len(line)>num_steps:
        return line[:num_steps]

    return line+[padding_token]*(num_steps-len(line))

# print(truncate_pad(src_vocab[source[0]],10,src_vocab['<pad>']))
#[47, 4, 1, 1, 1, 1, 1, 1, 1, 1]


def build_array_nmt(lines,vocab,num_steps):
    lines=[vocab[l] for l in lines]
    lines=[l+[vocab['<eos>']] for l in lines]
    array=torch.tensor([truncate_pad(l,num_steps,vocab['<pad>']) for l in lines])
    '''虽然截断了但是还要计算原有长度'''
    valid_len=(array!=vocab['<pad>']).type(torch.int32).sum(1)
    '''
    如果元素 不是 <pad>（即是真实单词），结果为 True (相当于 1)。
    如果元素 是 <pad>（即填充位），结果为 False (相当于 0)。
    '''
    return array,valid_len


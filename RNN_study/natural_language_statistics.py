import random
import torch
from d2l import torch as d2l
from text_pre_process import load_corpus_time_machine,Vocab

tokens=load_corpus_time_machine()
corpus,vocab = load_corpus_time_machine()
print(vocab._token_freqs[:10]) ## _token_freqs是属性，token_freqs是方法


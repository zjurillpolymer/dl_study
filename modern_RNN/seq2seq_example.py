import collections
import math
import torch
from torch import nn
from d2l import torch as d2l



class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0,**kwargs):
        super(Seq2SeqEncoder,self).__init__(**kwargs)

        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)

    def forward(self, X, *args):
        X=self.embedding(X)
        X=X.permute(1,0,2)
        output,state=self.rnn(X)

        return output,state
encoder=Seq2SeqEncoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layers=2)
encoder.eval()
X=torch.zeros((4,7),dtype=torch.long)
output,state=encoder(X)
print(output.shape)

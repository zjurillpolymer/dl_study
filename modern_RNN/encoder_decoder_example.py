'''
这是一个接口，仅仅规定了框架
'''
from torch import nn

class Encoder(nn.Module):
    def __init__(self,**kwargs):
        super(Encoder,self).__init__(**kwargs)

    def forward(self,x): ## x通常是源序列
        '''
        这是一个抽象类。你不能直接实例化它，必须继承它并实现具体的 forward 逻辑（比如使用 LSTM, GRU, 或 Transformer Encoder）。
        '''
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self,**kwargs):
        super(Decoder,self).__init__(**kwargs)

    def init_state(self,enc_outputs,*args):

        raise NotImplementedError  ##接受Encoder的输出enc_outputs，并将其转为Decoder所需要的初始状态

    def forward(self,x,state):
        raise NotImplementedError

    '''
    整体架构
    '''
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,**kwargs):
        super(EncoderDecoder,self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,enc_X,dec_X,*args):
        enc_outputs = self.encoder(enc_X,*args)  ## 在简单的 RNN 中，这通常是最后一步的隐藏状态（Hidden State）。
        '''
        encoder最终的隐状态决定了decoder的初始隐状态？
        '''
        dec_state=self.decoder.init_state(enc_outputs,*args)
        return self.decoder(dec_X,dec_state)
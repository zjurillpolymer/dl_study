import torch
from torch import nn
from d2l import torch as d2l
from RNN_study import text_pre_process

batch_size,num_steps=32,35
train_iter,vocab=text_pre_process.load_data_time_machine(batch_size,num_steps)

def get_params(vocab_size,num_hiddens,device):
    num_inputs=num_outputs=vocab_size

    def normal(shape):
        return torch.rand(size=shape,device=device)*0.01

    def three():
        return (normal((num_inputs,num_hiddens)),normal((num_hiddens,num_hiddens)),torch.zeros(size=(num_hiddens,num_hiddens)))

    W_xz,W_hz,b_z=three()
    W_xr,W_hr,b_r=three()
    W_xh,W_hh,b_h=three()

    W_hq=normal((num_hiddens,num_outputs))
    b_q=torch.zeros(num_outputs,device=device)

    params=[W_xz,W_hz,b_z,W_xr,W_hr,b_r,W_hh,b_h,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),)
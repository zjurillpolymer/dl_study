import torch
import torch.nn.functional as F
from torch import nn


x=torch.arange(4)
# torch.save(x,'x-file')
# x2=torch.load('x-file')
# print(x2)

y=torch.zeros(4)
torch.save([x,y],'x-file')
x2,y2=torch.load('x-file')
print(x2)
print(y2)
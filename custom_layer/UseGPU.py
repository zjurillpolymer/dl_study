import torch
from torch import nn

# print(torch.cuda.device_count())

# def try_gpu(i=0): #@save
#     if torch.cuda.device_count() >= i + 1:
#         return torch.device(f'cuda:{i}')
#     return torch.device('cpu')
#
# def try_all_gpus():  # @save
#     devices = [torch.device(f'cuda:{i}')
#                for i in range(torch.cuda.device_count())]
#
#     return devices if devices else [torch.device('cpu')]
#
# X=torch.ones(2,3,device=try_gpu())
# print(X.device)
#
# Y=torch.ones(2,3,device=try_gpu(1))
# print(Y.device)

# print(torch.cuda.is_available())        # 应该返回 True
# print(torch.cuda.get_device_name(0))


device=torch.device('cuda')

x=torch.tensor([1,2,3,4])
x=x.to(device)

y=torch.tensor([5,6,7,8])
print(x+y.to(device))
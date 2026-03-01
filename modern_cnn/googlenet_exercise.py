import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

'''
Inception块由四条并⾏路径组成。前三条路径使⽤窗⼝⼤⼩为1 * 1、3 * 3和5 * 5的卷积层，
从不同空间⼤⼩中提取信息。中间的两条路径在输⼊上执⾏1 * 1卷积，以减少通道数，从⽽降低模型的复杂
性。第四条路径使⽤3 * 3最⼤汇聚层，然后使⽤1 * 1卷积层来改变通道数。这四条路径都使⽤合适的填充
来使输⼊与输出的⾼和宽⼀致，最后我们将每条线路的输出在通道维度上连结，并构成Inception块的输出。
'''
class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4,**kwargs):
        super(Inception,self).__init__(**kwargs)

        '''线路1，单1*1卷积层'''
        self.p1_1=nn.Conv2d(in_channels,c1,kernel_size=1)

        '''线路2，1*1卷积层后接3*3卷积层'''
        self.p2_1=nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.p2_2=nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)

        '''线路3，1*1卷积层后接5*5卷积层'''
        self.p3_1=nn.Conv2d(in_channels,c3[0],kernel_size=1)
        self.p3_2=nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)

        '''线路4，3*3最大汇聚层后接1*1卷积层'''
        self.p4_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2=nn.Conv2d(in_channels,c4,kernel_size=1)

    def forward(self,x):
        p1=F.relu(self.p1_1(x))
        p2=F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3=F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4=F.relu(self.p4_2(self.p4_1(x)))

        return torch.cat((p1,p2,p3,p4),1)
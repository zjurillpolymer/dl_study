import os
# 必须在导入任何可能使用 OpenMP 的库之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

def show_heatmaps(matrices,xlabel,ylabel,titles=None,figsize=(5,5),
                  cmap='Reds'):
    d2l.use_svg_display()
    num_rows,num_cols=matrices.shape[0],matrices.shape[1] ##行数和列数
    '''
    N (num_rows)：表示热力图网格有多少行。
    M (num_cols)：表示热力图网格有多少列。
    H, W：每个具体热力图矩阵的高度和宽度（例如 10x10）。
    '''
    fig,axes=d2l.plt.subplots(num_rows,num_cols,figsize=figsize,
                              sharex=True,sharey=True,squeeze=False)  ## axes是一个(num_rows,num_cols)大小的numpy array
    for i,(row_axes,row_matrices) in enumerate(zip(axes,matrices)):
        '''
        list(enumerate(['A', 'B', 'C']))
        # 输出: [(0, 'A'), (1, 'B'), (2, 'C')]
        So,the 'eunumerate' function has the function of 'range.
        '''
        '''
        解码以后，row_axes和row_matrices分别对应axes和matrices中的行'''
        for j,(ax,matrix) in enumerate(zip(row_axes,row_matrices)):
            pcm=ax.imshow(matrix.detach().numpy(),cmap=cmap)
            if i==num_rows-1:
                ax.set_xlabel(xlabel)
            if j==0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm,ax=axes,shrink=0.6)
    plt.show()
attention_weights=torch.eye(10).reshape((1,1,10,10))
'''eye音同I，即创造单位矩阵'''
show_heatmaps(attention_weights,xlabel='Keys',ylabel='Queries')
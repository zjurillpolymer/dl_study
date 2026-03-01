import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib.pyplot as plt
import numpy as np
from text_pre_process import load_corpus_time_machine


corpus,vocab=load_corpus_time_machine()

token_freqs=vocab._token_freqs

###print(token_freqs[:10])
###output:[('the', 2261), ('i', 1267), ('and', 1245), ('of', 1155), ('a', 816), ('to', 695), ('was', 552), ('in', 541), ('that', 443), ('my', 440)]
###均为元组
'''
print(ranks[:10])
[ 1  2  3  4  5  6  7  8  9 10]
即排名
'''

'''
print(freqs[:10])
[2261, 1267, 1245, 1155, 816, 695, 552, 541, 443, 440]
即出现频次前十的词汇的频率
'''


ranks=np.arange(1,len(token_freqs)+1)
freqs=[freq for token,freq in token_freqs]


'''
齐夫定律：词频 f 与排名 r 成反比

公式：f(r) ∝ 1/r

双对数坐标下：log(f) = -log(r) + C
              ↑ 斜率约为 -1 的直线
'''

plt.figure(figsize=(10,6))
plt.loglog(ranks,freqs,'b-',linewidth=1.5,label='Actual')
zipf_freqs = freqs[0] / ranks
plt.loglog(ranks, zipf_freqs, 'r--', linewidth=2, label='Zipf Law (slope=-1)')

# 6. 计算实际斜率（线性拟合）
log_ranks = np.log(ranks)
log_freqs = np.log(freqs)
slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
print(f"实际斜率：{slope:.4f}")  # 应该接近 -1

# 添加拟合线
fit_freqs = np.exp(intercept) * ranks ** slope
plt.loglog(ranks, fit_freqs, 'g-.', linewidth=2, label=f'Fit (slope={slope:.2f})')

# 7. 设置图表
plt.xlabel('Rank (log scale)', fontsize=12)
plt.ylabel('Frequency (log scale)', fontsize=12)
plt.title("Zipf's Law Verification - Time Machine Corpus", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


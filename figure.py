
import numpy as np
import matplotlib.pyplot as plt

# 加载55维时间序列数据
time_series = np.load('./data/data/data/test/E-1.npy')[:,19:]

# 假设 ranges 是异常范围的列表，例如：[[290, 390], [1540, 1575]]
ranges = [[550, 750], [2100, 2210]]

# 创建一个与时间序列长度相同的标签数组，初始值全部为0
labels = np.zeros(time_series.shape[0])

# 将异常范围内的值设置为1
for r in ranges:
    labels[r[0]:r[1]] = 1
    
print(labels.shape)

# 创建图形和轴
fig, axes = plt.subplots(time_series.shape[1], 1, figsize=(120, 22))  # 55个图，排成11行5列
axes = axes.flatten()

# 绘制每一维的时间序列
for i in range(time_series.shape[1]):
    ax = axes[i]
    ax.plot(time_series[:, i])
    
    # 标记异常范围
    for r in ranges:
        ax.axvspan(r[0], r[1], color='red', alpha=0.3)
    
    ax.set_title(f'Dimension {i+1}')

# 隐藏多余的轴
for j in range(time_series.shape[1], len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
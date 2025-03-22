import numpy as np
import pandas as pd
def make_labels(ids,data_path):
    df=pd.read_csv(data_path +"/labeled_anomalies.csv")
    df=df[df['chan_id']==ids]
    anomaly_sequences = eval(df['anomaly_sequences'].values[0])  # 将字符串转换为列表
    num_values = int(df['num_values'].values[0])
    # 生成标签
    labels = np.zeros(num_values)
    for seq in anomaly_sequences:
        start, end = seq
        labels[start-1:end] = 1  # Python索引从0开始，因此需要减1
    return labels

pred=np.load('pred.npy')
indices=np.where(pred==1)[0]
print(len(indices))
print(indices[0:500])


labels=make_labels("C-1",'./MSL&SMAP')
indices=np.where(labels==1)[0]
print(len(indices))
print(indices[-500:])
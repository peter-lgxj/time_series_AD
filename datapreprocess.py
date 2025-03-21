"""
Preprocess Criteo dataset. This dataset was used for the Display Advertising
Challenge (https://www.kaggle.com/c/criteo-display-ad-challenge).
"""
import os
import sys
import click
import random
import collections
import numpy as np
import pandas as pd
from dataset import CriteoDataset
from torch.utils.data import DataLoader
from torch.utils.data import sampler


# continous_features = [0]
# categorial_features = [1,55]


def preprocess(datadir, id,continous_features,categorial_features,batch_size,seq_length,step_size):
    train_data=np.load(datadir+'/train/'+id+'.npy')
    test_data=np.load(datadir+'/test/'+id+'.npy')

    print("训练数据:", train_data.shape)
    print("测试数据:", test_data.shape)
    
    all_data=np.concatenate((train_data,test_data),axis=0)
    # print(all_data.shape)
    # 计算每个通道的方差
    variances = np.var(all_data, axis=0)
    # print("每个通道的方差:", variances)
    
    # 删除方差为0的通道
    non_zero_variance_indices = np.where(variances != 0)[0]
    print("非零方差通道的索引:", non_zero_variance_indices)
    train_data = train_data[:, non_zero_variance_indices]
    test_data = test_data[:, non_zero_variance_indices]
    all_data = all_data[:, non_zero_variance_indices]
    print("all_data shape:", all_data.shape)
    
    count=np.sum(variances==0)
    categorial_features[1]=categorial_features[1]-count


    for feature in range(continous_features[0], continous_features[1]+1 if continous_features[1] == continous_features[0] else continous_features[1]):
        print("Standardizing_continous_feature:", feature)
        feature_data = all_data[:, feature]
        mean = np.mean(feature_data)
        std = np.std(feature_data)
        train_data[:, feature] = (train_data[:,feature] - mean) / std
        test_data[:, feature] = (test_data[:,feature] - mean) / std
        # all_data[:, feature] = (all_data[:,feature] - mean) / std
    
    # 统计离散维度的类别数目
    categorial_counts = [len(np.unique(all_data[:, feature])) for feature in range(categorial_features[0], categorial_features[1]+1 if categorial_features[1] == categorial_features[0] else categorial_features[1])]
    continous_count = [1 for _ in range(continous_features[0], continous_features[1]+1 if continous_features[1] == continous_features[0] else continous_features[1])]
    feature_counts = continous_count + categorial_counts
    print("Feature counts:", feature_counts)

    train_dataset=CriteoDataset(data=train_data, window_size=seq_length, step_size=step_size, train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    
    test_dataset=CriteoDataset(data=test_data, window_size=seq_length, step_size=step_size, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=5)
    
    return train_loader,test_loader,feature_counts,continous_features,categorial_features
    

if __name__ == "__main__":
    continous_features = [0,0]
    categorial_features = [1,55]
    id = "C-1"
    batch_size = 1
    seq_length = 3
    step_size = 1
    train_loader,test_loader,feature_counts =preprocess('./MSL&SMAP/data',id,continous_features,categorial_features,batch_size,seq_length,step_size)
    # print(feature_counts)




#for test 0923

#datadir = '../data/raw'
#outdir = '../data'
#dicts = CategoryDictGenerator(len(categorial_features))
#dicts.build(
#    os.path.join(datadir, 'train.txt'), categorial_features, cutoff=10)
#dict_sizes,dict_test = dicts.dicts_sizes()


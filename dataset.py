import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class CriteoDataset(Dataset):

    def __init__(self, data=None, window_size=5, step_size=1, train=True):
        self.train = train
        self.window_size = window_size
        self.step_size = step_size

        if self.train:
            self.data = data
        else:
            self.data = data

    def __getitem__(self, idx):
        actual_idx = idx * self.step_size
        if actual_idx + self.window_size > len(self.data):
            raise IndexError('Index out of range for sliding window')
        
        data_window = self.data[actual_idx:actual_idx + self.window_size, :]
        # target = self.data[actual_idx + self.window_size - 1, -1]
        
        Xi = torch.from_numpy(data_window.astype(np.float16)).unsqueeze(-1)
        Xv = torch.from_numpy(np.ones_like(data_window))
        return Xi, Xv

    def __len__(self):
        return (len(self.data) - self.window_size) // self.step_size + 1
# -*- coding: utf-8 -*-

"""
A pytorch implementation of DeepFM for rates prediction problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from time import time


class DeepFM(nn.Module):
    def __init__(self, feature_sizes, embedding_size=4,
                 hidden_dims=[32, 32], num_classes=10, dropout=[0.5, 0.5], 
                 verbose=False):

        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        # self.sigmoid = nn.ReLU()
        self.dtype = torch.float

#        self.fm_first_order_embeddings = nn.ModuleList(
#            [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
        fm_first_order_Linears = nn.ModuleList(
                [nn.Linear(feature_size, self.embedding_size) for feature_size in self.feature_sizes[:13]])
        fm_first_order_embeddings = nn.ModuleList(
                [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes[13:40]])
        self.fm_first_order_models = fm_first_order_Linears.extend(fm_first_order_embeddings)

#        self.fm_second_order_embeddings = nn.ModuleList(
#            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        fm_second_order_Linears = nn.ModuleList(
                [nn.Linear(feature_size, self.embedding_size) for feature_size in self.feature_sizes[:13]])
        fm_second_order_embeddings = nn.ModuleList(
                [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes[13:40]])
        self.fm_second_order_models = fm_second_order_Linears.extend(fm_second_order_embeddings)

        all_dims = [self.field_size * self.embedding_size] + \
            self.hidden_dims + [self.num_classes]
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_'+str(i),
                    nn.Linear(all_dims[i-1], all_dims[i]))
            # nn.init.kaiming_normal_(self.fc1.weight)
            setattr(self, 'batchNorm_' + str(i),
                    nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_'+str(i),
                    nn.Dropout(dropout[i-1]))

    def forward(self, Xi, Xv):
        print(Xi)
        print(Xv)
        emb = self.fm_first_order_models[20]
#        print(Xi.size())
        for num in Xi[:, 20, :][0]:
            if num > self.feature_sizes[20]:
                print("index out")

#        fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_first_order_models)]
#        fm_first_order_emb_arr = [(emb(Xi[:, i, :]) * Xv[:, i])  for i, emb in enumerate(self.fm_first_order_models)]
        fm_first_order_emb_arr = []
        for i, emb in enumerate(self.fm_first_order_models):
            if i <=12:
                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.float)
                fm_first_order_emb_arr.append((torch.sum(emb(Xi_tem).unsqueeze(1), 1).t() * Xv[:, i]).t())
            else:
                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.long)
                fm_first_order_emb_arr.append((torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]).t())
#        print("successful")      
#        print(len(fm_first_order_emb_arr))
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
#        fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_second_order_models)]
        # fm_second_order_emb_arr = [(emb(Xi[:, i]) * Xv[:, i]) for i, emb in enumerate(self.fm_second_order_embeddings)]
        fm_second_order_emb_arr = []
        for i, emb in enumerate(self.fm_second_order_models):
            if i <=12:
                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.float)
                fm_second_order_emb_arr.append((torch.sum(emb(Xi_tem).unsqueeze(1), 1).t() * Xv[:, i]).t())
            else:
                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.long)
                fm_second_order_emb_arr.append((torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]).t())
                
        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * \
            fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [
            item*item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(
            fm_second_order_emb_square)  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square -
                           fm_second_order_emb_square_sum) * 0.5

#        print(len(fm_second_order_emb_arr))
#        print(torch.cat(fm_second_order_emb_arr, 1).shape)
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)
#            print("successful") 

#        print("1",torch.sum(fm_first_order, 1).shape)
#        print("2",torch.sum(fm_second_order, 1).shape)
#        print("deep",torch.sum(deep_out, 1).shape)
#        print("bias",bias.shape)
        bias = torch.nn.Parameter(torch.randn(Xi.size(0)))
        total_sum = torch.sum(fm_first_order, 1) + \
                    torch.sum(fm_second_order, 1) + \
                    torch.sum(deep_out, 1) + bias
        # total_sum = self.sigmoid(total_sum)
        return total_sum


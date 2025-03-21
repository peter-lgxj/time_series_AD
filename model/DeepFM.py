# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class DeepFMEmbedding(nn.Module):
    def __init__(self, feature_sizes,continous_features,categorial_features, embedding_size=4):
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.continous_features=continous_features
        self.categorial_features=categorial_features

        fm_first_order_Linears = nn.ModuleList(
            [nn.Linear(feature_size, self.embedding_size) for feature_size in self.feature_sizes[continous_features[0]:continous_features[1]+1 if continous_features[1] == continous_features[0] else continous_features[1]]])
        fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes[categorial_features[0]:categorial_features[1]+1 if categorial_features[1] == categorial_features[0] else categorial_features[1]]])
        self.fm_first_order_models = fm_first_order_Linears.extend(fm_first_order_embeddings)

        fm_second_order_Linears = nn.ModuleList(
            [nn.Linear(feature_size, self.embedding_size) for feature_size in self.feature_sizes[continous_features[0]:continous_features[1]+1 if continous_features[1] == continous_features[0] else continous_features[1]]])
        fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes[categorial_features[0]: categorial_features[1]+1 if categorial_features[1] == categorial_features[0] else categorial_features[1]]])
        self.fm_second_order_models = fm_second_order_Linears.extend(fm_second_order_embeddings)

    def forward(self, Xi, Xv):
        B, L, C, D = Xi.size()
        Xi = Xi.view(B* L , C, D)
        Xv = Xv.view(B* L , C)

        fm_first_order_emb_arr = []
        for i, emb in enumerate(self.fm_first_order_models):
            if i <= self.continous_features[1]:
                Xi_tem = Xi[:, i, :].to(dtype=torch.float)
                fm_first_order_emb_arr.append((torch.sum(emb(Xi_tem).unsqueeze(1), 1).t() * Xv[:, i]).t())
            else:
                Xi_tem = Xi[:, i, :].to(dtype=torch.long)
                fm_first_order_emb_arr.append((torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]).t())
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)

        fm_second_order_emb_arr = []
        for i, emb in enumerate(self.fm_second_order_models):
            if i <= self.continous_features[1]:
                Xi_tem = Xi[:, i, :].to(dtype=torch.float)
                fm_second_order_emb_arr.append((torch.sum(emb(Xi_tem).unsqueeze(1), 1).view(B * L, -1) * Xv[:, i].view(B * L, -1)).view(B * L, -1))
            else:
                Xi_tem = Xi[:, i, :].to(dtype=torch.long)
                fm_second_order_emb_arr.append((torch.sum(emb(Xi_tem), 1).view(B * L, -1) * Xv[:, i].view(B * L, -1)).view(B * L, -1))

        fm_first_order = fm_first_order.view(B, L, -1)
        # fm_second_order_emb_arr = [emb.view(B, L, -1) for emb in fm_second_order_emb_arr]

        return fm_first_order, fm_second_order_emb_arr


class DeepFM(nn.Module):
    def __init__(self, feature_sizes, bias_size, embedding_size=4,
                 hidden_dims=[32, 32], num_classes=10, dropout=[0.5, 0.5], 
                 verbose=False):

        super().__init__()
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        all_dims = [len(feature_sizes) * embedding_size] + hidden_dims + [num_classes]
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(all_dims[i-1], all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i-1]))
        self.bias = torch.nn.Parameter(torch.randn(bias_size))

    def forward(self, fm_first_order, fm_second_order_emb_arr):

        B, L, C = fm_first_order.size()
        fm_first_order = fm_first_order.view(B* L , C)
        
        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb
        fm_second_order_emb_square = [item * item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5

        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)

        total_sum = torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + torch.sum(deep_out, 1) + self.bias
        return total_sum





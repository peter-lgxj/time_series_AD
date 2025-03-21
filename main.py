# from model.DeepFM import DeepFMEmbedding,DeepFM
from model.MyModel import MyModel
from datapreprocess import preprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import os



class EXP_AD(object):
    def __init__(self, datapath,id,config):
        self.id = id
        self.config = config
        self.datapath = datapath
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        self.step_size = config.step_size
        self.continous_features = config.continous_features
        self.categorial_features = config.categorial_features
        self.embedding_size = config.embedding_size

        self.depth = config.depth
        self.heads = config.heads
        self.mlp_dim = config.mlp_dim
        self.dim_head = config.dim_head
        self.dropout = config.dropout
        
        self.train_loader, self.test_loader, self.feature_counts, self.continous_features, self.categorial_features = preprocess(self.datapath, self.id, self.continous_features, self.categorial_features, self.batch_size, self.seq_length, self.step_size)
        
        self.model = MyModel(feature_counts=self.feature_counts, 
                             continous_features=self.continous_features, 
                             categorial_features=self.categorial_features, 
                             embedding_size=self.embedding_size, 
                             num_df=len(self.feature_counts)*self.embedding_size,
                             batch_size=self.batch_size, 
                             seq_length=self.seq_length, 
                             hidden_dim=len(self.feature_counts)*self.embedding_size, 
                             depth=self.depth, 
                             heads=self.heads, 
                             mlp_dim=self.mlp_dim, 
                             dim_head=self.dim_head, 
                             dropout=self.dropout)
        
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss.to(self.device)
        # self.optimizer.to(self.device)
        # self.train()
        # self.test()
        
    def train(self):
        self.model.train()
        for i, (Xi, Xv) in enumerate(self.train_loader):
            Xi, Xv = Xi.to(self.device), Xv.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(Xi, Xv)
            base=Xi.squeeze(-1)
            loss = self.loss(out, base)
            loss.backward()
            self.optimizer.step()
            if i % 100 == 0:
                print(i, loss.item())
                
    def test(self):
        self.model.eval()
        with torch.no_grad():
            for i, (Xi, Xv) in enumerate(self.test_loader):
                Xi, Xv = Xi.to(self.device), Xv.to(self.device)
                out = self.model(Xi, Xv)
                loss = self.loss(out, Xi)
                if i % 100 == 0:
                    print(i, loss.item())
                    
    def predict(self,Xi,Xv):
        self.model.eval()
        with torch.no_grad():
            Xi, Xv = Xi.to(self.device), Xv.to(self.device)
            out = self.model(Xi, Xv)
            return out
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        



if __name__ == "__main__":
    from datapreprocess import preprocess
    continous_features = [0,0]
    categorial_features = [1,55]
    id = "C-1"
        
    parser = argparse.ArgumentParser(description='Anomaly Detection')
    parser.add_argument('--datapath', type=str, default='./MSL&SMAP/data', help='data path')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size')
    parser.add_argument('--seq_length', type=int, default=50, help='sequence length')
    parser.add_argument('--step_size', type=int, default=1, help='step size')
    parser.add_argument('--continous_features', type=list, default=continous_features, help='continous features')
    parser.add_argument('--categorial_features', type=list, default=categorial_features, help='categorial features')
    parser.add_argument('--embedding_size', type=int, default=4, help='embedding size')
    

    parser.add_argument('--depth', type=int, default=6, help='number of transformer layers')
    parser.add_argument('--heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--mlp_dim', type=int, default=512, help='MLP dimension')
    parser.add_argument('--dim_head', type=int, default=64, help='dimension of each attention head')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--save_path', type=str, default='model.pth', help='model save path')
    parser.add_argument('--load_path', type=str, default='model.pth', help='model load path')
    
    args = parser.parse_args()
    
    exp=EXP_AD(args.datapath,id,args)
    exp.train()
    exp.test()
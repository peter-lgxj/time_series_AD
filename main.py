# from model.DeepFM import DeepFMEmbedding,DeepFM
from model.MyModel import MyModel
from datapreprocess import preprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import tqdm
import pandas as pd



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
        
        self.model_save_path= config.model_save_path
        self.dataset = config.dataset
        
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
        for epoch in range(self.config.epochs):
            print("Epoch: ", epoch)
            epoch_loss = 0.0
            for i, (Xi, Xv) in enumerate(self.train_loader):
                Xi, Xv = Xi.to(self.device), Xv.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(Xi, Xv)
                base=Xi.squeeze(-1)
                loss = self.loss(out, base)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                if i % 100 == 0:
                    print(f"Batch {i}, Loss: {loss.item()}")
                print(f"Epoch {epoch}, Total Loss: {epoch_loss}")  # 打印当前epoch的总loss
                torch.save(self.model.state_dict(), os.path.join(str(self.model_save_path), str(self.dataset) + self.id +'_checkpoint.pth'))
                

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + self.id+ '_checkpoint.pth')))
        self.model.eval()
        temperature = 50
        self.standard=torch.eye(self.win_size).to(self.device)
        print("======================TEST MODE======================")

        self.evaluation=nn.MSELoss(reduction='mean')
        # (3) evaluation on the test set
        test_labels = []
        # attns = []
        norm_out=[]
        abnorm_out=[]
        test_labels=self.make_labels('MSL',self.id, self.datapath)
        for i, (Xi, Xv) in enumerate(self.test_loader):
            Xi, Xv = Xi.to(self.device), Xv.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(Xi, Xv)
            base=Xi.squeeze(-1)
            loss = self.loss(out, base)
            norm_out.append(loss.detach().cpu().numpy())
            # abnorm_out.append(out[:,1].detach().cpu().numpy())
        norm_out=np.concatenate(norm_out, axis=0)
        # abnorm_out=np.concatenate(abnorm_out, axis=0)
        # attns=np.concatenate(attns, axis=0)
        # print(attns.shape)
        
        ma, mp, mr, mf=0.0,0.0,0.0,0.0
        nums = 0.0
        while nums < self.anormly_ratio:
            nums += 0.1
            norm_thresh = np.percentile(norm_out,nums)
            abnorm_thresh = np.percentile(abnorm_out,nums)
            normp = (norm_out < norm_thresh).astype(int)
            abnormp=(norm_out > abnorm_thresh).astype(int)
            pred = np.bitwise_and(normp, abnormp)
            if 1 not in pred:
                pred=normp
            preds=[0 for i in range(len(test_labels))]
            for index,value in enumerate(pred):
                if value==1:
                    # if pred[index+1]==1:
                    for i in range(self.win_size//2):
                        preds[index*self.step+i]=1
            # print(len(preds))
            gt = test_labels.astype(int)
            if self.pa:
                self.pa_stratge(gt,preds)
            accuracy, precision, recall, f_score=self.compute_f1(gt,preds)
            if f_score>mf:
                ma,mp,mr,mf=accuracy, precision, recall, f_score
                print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                    accuracy, precision,recall, f_score))
                print("n_Threshold :", norm_thresh)
                # print("abn_Threshold :", abnorm_thresh)
                print("pred:   ", len(preds))
                print("gt:     ", gt.shape)
                np.save("pred.npy",preds)
                # np.save("attns.npy",attns)
        return ma, mp, mr, mf

    def make_labels(self,dataset,ids,data_path):
        if dataset=='MSL' or 'SMAP':
            df=pd.read_csv(data_path +"/labeled_anomalies.csv")
            df=df[df['chan_id']==ids]
            anomaly_sequences = eval(df['anomaly_sequences'].values[0])  # 将字符串转换为列表
            num_values = int(df['num_values'].values[0])
            # 生成标签
            labels = np.zeros(num_values)
            for seq in anomaly_sequences:
                start, end = seq
                labels[start:end] = 1  
            return labels
        elif dataset=='SMD':
            labels = np.loadtxt(data_path + "/test_label/machine-"+ids+".txt", delimiter=',')
            return labels
        
    def compute_f1(self,gt,preds):
        pred = np.array(preds)
        gt = np.array(gt)
        # print("pred: ", pred.shape)
        # print("gt:   ", gt.shape)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        return accuracy, precision, recall, f_score
        
    def pa_stratge(self,gt,preds):
        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and preds[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if preds[j] == 0:
                            preds[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if preds[j] == 0:
                            preds[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                preds[i] = 1
                


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
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--dataset', type=str, default='MSL', help='dataset name')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints', help='model save path')
    # parser.add_argument('--id', type=str, default=id, help='id')
    
    args = parser.parse_args()
    
    exp=EXP_AD(args.datapath,id,args)
    exp.train()
    exp.test()
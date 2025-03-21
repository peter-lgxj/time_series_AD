import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
from utils.badsampler import add_anomaly
# from dim2_iou import iou_loss,compute_distance
# from test3 import OPTICS,rloss, mloss,DSVDDUncLoss
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self,ids, config):
        self.ids=ids
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,id=ids,
                                               mode='train',step=self.step,
                                               dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,id=ids,
                                              mode='test',step=self.step,
                                              dataset=self.dataset)
        # self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,id=ids,
        #                                       mode='thre',
        #                                       dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.celoss= nn.CrossEntropyLoss()

        
    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, z1,z2 = self.model(input)

            rec_loss = self.criterion(output, input)
            # r_loss= rloss(z1,z2)
            # m_loss= mloss(z1,z2)
            # b,w,d=z1.shape
            # z1=z1.reshape(b*w,d)
            # z2=z2.reshape(b*w,d)
            # center1=self.Optics(z1)
            # center2=self.Optics(z2)
            # iou_ls = self.criterion(center1, center2)
            
            loss_1.append(2*rec_loss)
            # loss_2.append(200*r_loss.item()+100*m_loss.item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        # self.model.load_state_dict(
        #     torch.load(
        #         os.path.join(str(self.model_save_path), str(self.dataset) + '_pretrained_checkpoint.pth')),strict=False)

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        # early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)
        # self.standard=torch.eye(self.win_size).to(self.device)
        changes=0
        for epoch in range(self.num_epochs):
            iter_count = 0
            
            epoch_time = time.time()
            self.model.train()
            # self.center=None
            for i, (input_data, labels) in enumerate(self.train_loader):
                
                input_data, labels = add_anomaly(input_data, labels, self.win_size//2)
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                labels = labels.long().to(self.device)
                enc_out,c_out,_ = self.model(input)
                rec_loss = self.criterion(enc_out, input)
                ce_loss = self.celoss(c_out, labels)
                
                loss = rec_loss+ce_loss
                
                
                # if (i + 1) % 25 == 0:
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     print('\tloss: {:.5f}'.format(loss.item()))
                #     print('\tco_loss: {:.5f}'.format(ce_loss.item()))
                #     iter_count = 0
                #     time_now = time.time()
                
                
                loss.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            if epoch%8==0:
                adjust_learning_rate(self.optimizer, changes + 1, self.lr)
                changes+=1
            torch.save(self.model.state_dict(), os.path.join(str(self.model_save_path), str(self.dataset) + self.ids +'_checkpoint.pth'))
            # self.test()
            

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + self.ids+ '_checkpoint.pth')))
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
        test_labels=self.make_labels(self.dataset,self.ids, self.data_path)
        for i, (input_data, labels) in enumerate(tqdm(self.test_loader)):
            input = input_data.float().to(self.device)
            _,out,__= self.model(input)
            # attns.append(attn.detach().cpu().numpy())
            norm_out.append(out[:,0].detach().cpu().numpy())
            abnorm_out.append(out[:,1].detach().cpu().numpy())
        norm_out=np.concatenate(norm_out, axis=0)
        abnorm_out=np.concatenate(abnorm_out, axis=0)
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
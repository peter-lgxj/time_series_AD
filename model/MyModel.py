import torch
from torch import nn
from einops import rearrange

from model.PositionalEncoding import PositionalEncoding
from model.transformer import Transformer
from model.DeepFM import DeepFM, DeepFMEmbedding


class TFUnit(nn.Module):
    def __init__(self, hidden_dim, num_patches, depth, heads, mlp_dim, dim_head, dropout):
        super().__init__()
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_dim, dropout=dropout, max_len=num_patches)
        self.transformer = Transformer(
            hidden_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        b, n, _ = x.shape
        x_embeddings = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        x_embeddings = self.transformer(x_embeddings)
        return x_embeddings


class MyModel(nn.Module):
    def __init__(self, feature_counts, continous_features, categorial_features,embedding_size,num_df,
                 batch_size,seq_length,hidden_dim, num_patches, depth, heads, mlp_dim, dim_head, dropout):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.dfemb = DeepFMEmbedding(feature_counts, continous_features, categorial_features,embedding_size)
        self.deepfm = nn.ModuleList([DeepFM(feature_counts,batch_size*seq_length) for _ in range(num_df)])
        # self.deepfm=DeepFM(feature_counts,batch_size*seq_length)
        self.vit_unit = TFUnit(hidden_dim, num_patches, depth, heads, mlp_dim, dim_head, dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, xi,xv):
        fm_first_order, fm_second_order_emb_arr = self.dfemb(xi,xv)
        
        fm_first_order = fm_first_order.float()  # 确保 fm_first_order 是 float 类型
        fm_second_order_emb_arr = [item.float() for item in fm_second_order_emb_arr]  # 确保 fm_second_order_emb_arr 中的每个元素都是 float 类型
        
        # df_out = self.deepfm(fm_first_order, fm_second_order_emb_arr).view(self.batch_size, self.seq_length)
        df_out=[]
        for df in self.deepfm:
            df_out.append(df(fm_first_order, fm_second_order_emb_arr))
        df_out=torch.stack(df_out,dim=1)
        
        
        ts_out = self.vit_unit(fm_first_order)
        # ts_out = self.fc(ts_out).squeeze(-1)
        return df_out, ts_out




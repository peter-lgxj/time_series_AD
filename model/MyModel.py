import torch
from torch import nn
from einops import repeat, pack, unpack
from huggingface_hub import PyTorchModelHubMixin


from positional_encoding import PositionalEncoding
from transformer import Transformer


class ViTUnit(nn.Module):
    def __init__(self, hidden_dim, num_patches, depth, heads, mlp_dim, dim_head, dropout, device):
        super().__init__()
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_dim, dropout=dropout, max_len=num_patches+1)
        self.cls_token = nn.Parameter(torch.randn(hidden_dim).to(device))
        self.transformer = Transformer(
            hidden_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        x_embeddings, ps = pack([cls_tokens, x], 'b * d')
        x_embeddings = self.pos_encoder(
            x_embeddings.transpose(0, 1)).transpose(0, 1)
        x_embeddings = self.transformer(x_embeddings)
        cls_tokens, _ = unpack(x_embeddings, ps, 'b * d')
        return cls_tokens.reshape(cls_tokens.shape[0], -1)

if __name__ == '__main__':
    model = ViTUnit(512, 16, 6, 8, 1024, 64, 0.1, 'cpu')
    data= torch.randn(8, 16, 512).to('cpu')
    print(model(data).shape)
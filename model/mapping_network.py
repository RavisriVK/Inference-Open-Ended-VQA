import torch
import torch.nn as nn
import torch.nn.functional as nnf

class MappingNetwork(nn.Module):
    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.input_seq_length, -1)

        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((prefix, x), dim=-2)

        out = self.transformer(prefix)[:, :-self.input_seq_length]
        return out

    def __init__(self, dim_input_seq: int, dim_embedding: int, prefix_length: int, input_seq_length: int, num_layers: int = 8):
        super(MappingNetwork, self).__init__()
        self.input_seq_length = input_seq_length
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding,
            nhead=8,
            dim_feedforward=2*dim_embedding,
            dropout=0.0,
            activation=nnf.relu,
            batch_first=True,
            norm_first=True,
        )
        encoder_norm = nn.LayerNorm(dim_embedding)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
        
        self.linear = nn.Linear(dim_input_seq, dim_embedding)

        self.prefix_length = prefix_length
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)
import torch
from torch import nn
from math import log

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        pos_encoding = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # Position formula 
        # for even:  sin(pos/ (10000)^(2i/d_model))  
        # for odd: cos(pos/ (10000)^(2i/d_model))
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000) / d_model)) 
        pos_encoding[:, 0::2] = torch.sin(pos * denominator)
        pos_encoding[:, 1::2] = torch.cos(pos * denominator)
        return self.register_buffer('pos_encoding', pos_encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.pos_encoding[:, :x.size(1)]
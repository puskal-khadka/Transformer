import torch
from torch import nn
from math import sqrt


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head_count):
        super().__init__()    
        self.d_model = d_model
        self.head_count = head_count
        self.d_head = d_model // head_count
        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.out_layer = nn.Linear(d_model, d_model)


    def forward(self, q, k, v, mask=None):

        query = self.q_layer(q)
        batch_size, seq_length, d_model = query.size()
        query = query.reshape(batch_size, seq_length, self.head_count, self.d_head).transpose(1,2)

        key = self.k_layer(k)
        batch_size, seq_length, d_model = key.size()
        key = key.reshape(batch_size, seq_length, self.head_count, self.d_head).transpose(1,2)

        value = self.v_layer(v)
        batch_size, seq_length, d_model = value.size()
        value = value.reshape(batch_size, seq_length, self.head_count, self.d_head).transpose(1,2)

        attention_score = torch.matmul(query, key.transpose(-2, -1)) / sqrt(self.d_head)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e10)
        attention_score =  torch.softmax(attention_score, dim =-1)
        attention_score = torch.matmul(attention_score, value)
        batch_size, h_count, seq_length, d_head = attention_score.size()

        concat = attention_score.transpose(1,2).reshape(batch_size, seq_length, self.head_count * self.d_head) #=self.d_model

        return self.out_layer(concat)
    


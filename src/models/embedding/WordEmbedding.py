from torch import nn
from math import sqrt

class WordEmbedding(nn.Module):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    return  self.embedding(x) * sqrt(self.d_model)

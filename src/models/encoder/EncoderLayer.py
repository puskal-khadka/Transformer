from torch import nn
from src.models.multihead_attention import MultiHeadAttention
from src.models.feed_forward import PositionWiseFeedForwardLayer

class EncoderLayer(nn.Module):
    def __init__(self, d_model, head_count, d_ff, dropout=0.1):
        super().__init__()

        self.mult_head_attention = MultiHeadAttention(d_model, head_count)
        self.layer_norm_1 = nn.LayerNorm(d_model)

        self.feed_forward_layer = PositionWiseFeedForwardLayer(d_model, d_ff)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        # multi head attention & layer normalization => Fully connected layer & layer normalization  => encoder output
        attention = self.mult_head_attention(x, x, x, mask)
        x = self.layer_norm_1(x + self.dropout(attention))

        feed_forward = self.feed_forward_layer(x)
        return self.layer_norm_2(x + self.dropout(feed_forward))

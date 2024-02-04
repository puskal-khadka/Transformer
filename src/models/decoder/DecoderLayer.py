from torch import nn 
from src.models.feed_forward import PositionWiseFeedForwardLayer
from src.models.multihead_attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, head_count, d_ff, dropout=0.1):
        super().__init__()
        self.masked_multi_head_attetion = MultiHeadAttention(d_model, head_count)
        self.layer_norm_1 = nn.LayerNorm(d_model)

        self.cross_multi_head_attention = MultiHeadAttention(d_model, head_count)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.feed_forward_layer = PositionWiseFeedForwardLayer(d_model, d_ff)
        self.layer_norm_3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, context_summary, src_mask, target_mask):
        # cross_multi_head_attention (input:  output of encoder as (k,v) + masked_multi_head as (q)) & normalization  =>  FC layer  & normalization  => output

        masked_mha_out = self.masked_multi_head_attetion(x, x, x, target_mask)
        x = self.layer_norm_1(x + self.dropout(masked_mha_out))

    
        mha_out = self.cross_multi_head_attention(x, context_summary, context_summary, src_mask)
        x = self.layer_norm_2(x + self.dropout(mha_out))

        ff_out = self.feed_forward_layer(x)
        x = self.layer_norm_3(x + self.dropout(ff_out))
        return x
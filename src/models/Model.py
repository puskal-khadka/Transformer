from torch import nn
from src.models import embedding, encoder, decoder
from utils.Mask import maskGenerator

class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model, d_ff, stack, head_count, max_seq_len, dropout):
        super().__init__()
        self.source_embedding = embedding.WordEmbedding(source_vocab_size, d_model)
        self.target_embedding = embedding.WordEmbedding(target_vocab_size, d_model)

        self.positional_encoding = embedding.PositionalEncoding(d_model, max_seq_len)
         
        enc_layers = [ encoder.EncoderLayer(d_model, head_count, d_ff, dropout) for i in range(stack) ]
        dec_layers = [ decoder.DecoderLayer(d_model, head_count, d_ff, dropout) for i in range(stack) ]
        self.encoder_layers = nn.ModuleList(enc_layers)
        self.decoder_layers = nn.ModuleList(dec_layers)

        self.transformer_output = nn.Linear(d_model, target_vocab_size)

        self.dropout = nn.Dropout(dropout)


    def forward(self, src, target):
        src_mask, target_mask = maskGenerator(src, target)
        
        source_embedding = self.positional_encoding(self.source_embedding(src))
        source_embedding = self.dropout(source_embedding)

        target_embedding = self.positional_encoding(self.target_embedding(target))
        target_embedding = self.dropout(target_embedding)
        
        encoder_out = source_embedding
        for layer in self.encoder_layers:
            encoder_out =  layer(encoder_out, src_mask)

        decoder_out = target_embedding

        for layer in self.decoder_layers:
            decoder_out = layer(decoder_out, encoder_out, src_mask, target_mask)

        transformer_output = self.transformer_output(decoder_out)
        return transformer_output
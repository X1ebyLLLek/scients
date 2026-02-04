import torch
import torch.nn as nn
import math
from config import Config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Make pe a buffer, not a parameter, and transpose for broadcasting (1, max_len, d_model)
        self.register_buffer('pe', pe.transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, embed_size)
        # PE shape: (1, max_len, embed_size)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# This architecture is a prototype based on the report's recommendations
# BERT-style architecture (Bidirectional)
class TransformerPredictor(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.embed_size = embed_size
        # vocab_size includes padding(0), events(1..V), and mask(V+1). 
        # So overall embedding size needs to be V+2.
        # But wait, main.py passes `vocab_size` which comes from preprocessing.
        # Preprocessing: vocab_size = len(classes) + 1. 0 is PAD.
        # For MASK, we need one more.
        # Let's trust the upstream to pass the correct (extended) vocab_size,
        # OR we just allocate enough.
        # Let's allocate +2 just in case.
        self.embedding = nn.Embedding(vocab_size + 2, embed_size, padding_idx=0)
        
        self.pos_encoder = PositionalEncoding(embed_size, dropout,
                                              max_len=Config.MAX_SEQ_LEN + 1)
        
        # Batch_first=True is important.
        # Removing causal mask is default for TransformerEncoder (it only uses src_key_padding_mask).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=num_heads, batch_first=True, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer to predict token at each position
        self.fc = nn.Linear(self.embed_size, vocab_size + 2)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        # src: (B, L)
        # mask: (B, L)
        
        src = self.embedding(src) * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        
        # Bidirectional encoding (no causal mask)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        # output: (B, L, E)
        
        # Predict logits for EVERY token
        return self.fc(output) # (B, L, V)

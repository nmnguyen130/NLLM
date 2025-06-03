import math
from typing import Optional
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
    
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.d_k = d_model // heads
        self.heads = heads

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = q.shape

        # Ánh xạ q, k, v qua các linear layer
        q = self.query(q).view(batch, seq_len, self.heads, self.d_k).transpose(1, 2)
        k = self.key(k).view(batch, seq_len, self.heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch, seq_len, self.heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))  # True che, False ko che
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out(context)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.norm1(x)
        x = x + self.dropout(self.self_attn(x_norm, x_norm, x_norm, src_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.norm1(x)
        x = x + self.dropout(self.self_attn(x_norm, x_norm, x_norm, tgt_mask))
        x_norm = self.norm2(x)
        x = x + self.dropout(self.cross_attn(x_norm, enc_output, enc_output, src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 src_seq_len: int, tgt_seq_len: int, d_model: int = 512, 
                 layers: int = 6, heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        # Create the embedding layers
        self.src_embed = InputEmbedding(d_model, src_vocab_size)
        self.tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

        # Create the positional encoding layers
        self.src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        self.tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)
        ])
        self.proj = nn.Linear(d_model, tgt_vocab_size)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src = self.src_pos(self.src_embed(src))
        for layer in self.encoder:
            src = layer(src, src_mask)
        return src

    def decode(self, tgt: torch.Tensor, enc_output: torch.Tensor, 
              src_mask: Optional[torch.Tensor] = None, 
              tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt = self.tgt_pos(self.tgt_embed(tgt))
        for layer in self.decoder:
            tgt = layer(tgt, enc_output, src_mask, tgt_mask)
        return tgt
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.proj(dec_output)
import torch 
import torch.nn as nn
from typing import Optional, Tuple, Type
from .attention.attention import SingleHeadAttention
from .embedding.tokenEmbedding import TokenEmbedding
from .embedding.positionEmbedding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_len: int,
                 attn: Type[nn.Module]
                ):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(max_len, d_model)

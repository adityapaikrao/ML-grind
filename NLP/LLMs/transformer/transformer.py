import torch 
import torch.nn as nn
from typing import Optional, Tuple, Type
from .attention.attention import SingleHeadAttention
        

class Transformer(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_len: int,
                 attn: Type[nn.Module]
                ):
        pass
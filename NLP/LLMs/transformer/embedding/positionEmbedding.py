import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        """
        TODO: implement Sinsusoidal Position Embedding
        """
        pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: implement Sinsusoidal Position Embedding
        """
        return x + x.indices()
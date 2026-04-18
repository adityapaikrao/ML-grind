import torch 
import torch.nn as nn
from typing import Type, Optional
from .feedforward import FeedForward

class TransformerEncoderBlock(nn.Module):
    """
    A single Transformer block consisting of multi-head attention and a feed-forward network.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        d_model (int): The number of expected features in the input.
        num_heads (int): The number of heads in the multiheadattention models.
        d_ff (int): The dimension of the feedforward network model.
        max_len (int): The maximum sequence length.
        attn (Type[nn.Module]): Multi-head attention layer class.
    """
    def __init__(self, 
                 attn: Type[nn.Module],
                 d_model: int,
                 d_ff: Optional[int] = None,
                 num_heads: int = 1
                ):
        super().__init__()


        self.attention = attn(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x_BTD: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x_BTD (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model).
        """

        # Attention Block
        t_BTD = self.layer_norm1(x_BTD)
        t_BTD = self.attention(t_BTD)
        x_BTD = x_BTD + t_BTD

        # FeedForward Block
        t_BTD = self.layer_norm2(x_BTD)
        t_BTD = self.feed_forward(t_BTD)
        x_BTD = x_BTD + t_BTD

        return x_BTD
        


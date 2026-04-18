import torch
import torch.nn as nn
from typing import Optional

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network as described in the original Transformer paper.
    Consists of two linear transformations with a ReLU activation in between.
    """
    def __init__(self, 
                 d_model: int,
                 d_ff: Optional[int] = None,
                 dropout: float = 0.1
                ):
        """
        Initializes the FeedForward network.

        Args:
            d_model (int): The dimensionality of the input and output.
            d_ff (Optional[int]): The dimensionality of the inner layer. Defaults to 4 * d_model.
            dropout (float): Dropout probability. Defaults to 0.1.
        """
        super().__init__()

        d_ff = 4 * d_model if not d_ff else d_ff

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()
    
    def forward(self, x_BTD: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FeedForward network.

        Args:
            x_BTD (torch.Tensor): Input tensor of shape (Batch, Time, Dimension).

        Returns:
            torch.Tensor: Output tensor after linear layers, activation, and dropout.
        """
        t_BTDff = self.activation(self.fc1(x_BTD))
        t_BTDff = self.dropout(t_BTDff)

        t_BTD = self.fc2(t_BTDff)

        return t_BTD
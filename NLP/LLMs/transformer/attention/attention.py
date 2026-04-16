import torch 
import torch.nn as nn
from typing import Optional

class SingleHeadAttention(nn.Module):
    """
    Implementation of Scaled Dot-Product Attention for a single head.
    """

    def __init__(
            self,
            d_model: int,
            d_k: Optional[int] = None,
            d_v: Optional[int] = None
    ):
        """
        Initializes the SingleHeadAttention layer.

        Args:
            d_model (int): The embedding dimension of the input tokens.
            d_k (int, optional): The dimension of the query and key vectors. Defaults to d_model.
            d_v (int, optional): The dimension of the value vectors. Defaults to d_model.
        """

        super().__init__()
        self.d_model = d_model
        self.d_k = self.d_model if not d_k else d_k
        self.d_v = self.d_model if not d_v else d_v

        # Linear projections for Query, Key, and Value
        self.Wq_DD = nn.Linear(self.d_model, self.d_k, bias=False)
        self.Wk_DD = nn.Linear(self.d_model, self.d_k, bias=False)
        self.Wv_DD = nn.Linear(self.d_model, self.d_v, bias=False)
    
    def forward(self, x_BTD: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the attention mechanism.

        Args:
            x_BTD (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_v).
        """
        # Project input to Query, Key, and Value spaces
        Q_BTDk = self.Wq_DD(x_BTD)
        K_BTDk = self.Wk_DD(x_BTD)
        V_BTDv = self.Wv_DD(x_BTD)

        # Calculate scaled dot-product scores: (Batch, Target_len, Source_len)
        scores_BTT = torch.einsum('btk, bsk -> bts', Q_BTDk, K_BTDk) / (self.d_k ** 0.5)
        
        # Normalize scores to get attention weights
        attn_BTT = torch.softmax(scores_BTT, dim=-1)

        # Apply attention weights to values
        out_BTDv = torch.einsum('bts, bsv -> btv', attn_BTT, V_BTDv)
        return out_BTDv
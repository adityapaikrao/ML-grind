import torch.nn as nn
import torch

class TokenEmbedding(nn.Module):
    """
    Standard token embedding layer for Transformer models.
    Converts input tokens into dense vectors of size d_model.
    """
    def __init__(
            self, 
            vocab_size: int, 
            d_model: int
        ):
        """
        Initializes the TokenEmbedding.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the embedding vectors.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding_VD = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the token embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded output tensor.
        """
        return self.embedding_VD(x) * (self.d_model ** 0.5)
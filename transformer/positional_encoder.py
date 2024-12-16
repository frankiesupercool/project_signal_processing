import torch
import torch.nn as nn
import numpy as np


class PositionalEncoder(nn.Module):
    """
    Sinusoidal Positional Encoding class.

    Args:
        d_model (int): Embedding dimension (output dimensionality, usually matches embed_dim).
        max_len (int): Maximum sequence length for precomputation.
        zero_pad (bool): If True, positions start from 1 with 0-padding for position 0.
            This can be useful when the first position corresponds to padding tokens or special tokens.
            TODO probably not needed
        scale (bool): If True, the positional encodings are scaled by the square root of the
            embedding dimension (sqrt(num_units)). This scaling can help stabilize the magnitude of values
            in the input to the model, aligning better with initialization schemes of Transformer models
            (e.g., in the original Vaswani et al. paper).
    """

    def __init__(self, d_model, max_len=1024, zero_pad=False, scale=False):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.zero_pad = zero_pad
        self.scale = scale

        # Precompute positional encodings
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2] if d_model % 2 != 0 else position * div_term)

        if zero_pad:
            pe[0, :] = 0  # Zero out first position if required

        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x):
        """
        Add positional encodings to input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Input tensor with positional encodings added.
        """
        seq_len = x.size(1)
        assert seq_len <= self.max_len, "Sequence length exceeds max_len. Increase max_len for longer sequences."

        # Slice the precomputed encodings to match the sequence length
        pe = self.pe[:, :seq_len, :]
        if self.scale:
            pe = pe * (self.d_model ** 0.5)

        return x + pe

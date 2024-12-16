import torch
import torch.nn as nn
import numpy as np


class PositionalEncoder(nn.Module):
    """
    Sinusoidal Positional Encoding class.

    Args:
        num_units (int): The output dimensionality (usually matches embed_dim).
        zero_pad (bool): If True, positions start from 1 with 0-padding for position 0.
            This can be useful when the first position corresponds to padding tokens or special tokens.
            TODO probably not needed
        scale (bool): If True, the positional encodings are scaled by the square root of the
            embedding dimension (sqrt(num_units)). This scaling can help stabilize the magnitude of values
            in the input to the model, aligning better with initialization schemes of Transformer models
            (e.g., in the original Vaswani et al. paper).
    """

    def __init__(self, num_units=768, zero_pad=True, scale=True):
        super(PositionalEncoder, self).__init__()
        self.num_units = num_units
        self.zero_pad = zero_pad
        self.scale = scale

    def forward(self, inputs):
        """
        Forward pass for the sinusoidal positional encoding.

        Args:
            inputs (Tensor): Tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            Tensor: The inputs tensor with added positional encodings.
        """
        _, seq_len, embed_dim = inputs.size()
        assert embed_dim == self.num_units, "num_units must match the embedding dimension of inputs."

        position = torch.arange(seq_len, dtype=torch.float32, device=inputs.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.num_units, 2, dtype=torch.float32, device=inputs.device) *
                             -(np.log(10000.0) / self.num_units))

        # Compute positional encoding
        pe = torch.zeros(seq_len, self.num_units, device=inputs.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)  # Shape: (1, seq_len, num_units)
        if self.scale:
            pe = pe * (self.num_units ** 0.5)

        if self.zero_pad:
            pe[:, 0, :] = 0  # Zero out the first position if zero_pad is True

        return inputs + pe

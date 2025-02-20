import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """
    module encodes modality-specific information by adding a learnable
    encoding vector to the input tensor. This helps the model distinguish between
    different modalities (e.g., audio, video).

    Args:
        embed_dim (int): The embedding dimension for the modality-specific encoding.
                         Default is 768.
    """

    def __init__(self, embed_dim=768):
        super(ModalityEncoder, self).__init__()
        # Learnable encoding vector of shape (1, 1, embed_dim).
        # - The first dimension (1) is for a single modality.
        # - The second dimension (1) allows for sequence-wise broadcasting.
        # - The third dimension (embed_dim) is the embedding size.
        self.modality_encoding = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.2, requires_grad=True)

    def forward(self, encoded_modality):
        """
        Args:
            encoded_modality (torch.Tensor): The input tensor of shape
                                             (batch_size, seq_len, embed_dim),
                                             representing encoded features for a given modality.

        Returns:
            torch.Tensor: The input tensor with the modality encoding added,
                          maintaining the same shape as the input.
        """
        # expand the modality encoding to match the input tensor's shape, add vector element-wise.
        return self.modality_encoding.expand_as(encoded_modality)

import torch
import torch.nn as nn

class ModalityEncoder(nn.Module):
    """
    Encodes modalities.
    """

    def __init__(self, modality_dim, embed_dim=768):
        super(ModalityEncoder, self).__init__()
        self.audio_fc = nn.Linear(modality_dim, embed_dim)

    def forward(self, encoded_modality):
        """
        Args:
            encoded_modality: Tensor of shape (batch_size, seq_len, modality_dim)
        Returns:
            Combined tensor of shape (batch_size, total_seq_len, embed_dim)
        """
        modality_embed = self.audio_fc(encoded_modality)  # (batch_size, seq_len, embed_dim)
        return modality_embed

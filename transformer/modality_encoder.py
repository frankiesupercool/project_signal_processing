import torch
import torch.nn as nn

class ModalityEncoder(nn.Module):
    """
    Encodes and combines audio and video modalities.
    """

    def __init__(self, audio_dim, video_dim, embed_dim=768):
        super(ModalityEncoder, self).__init__()
        self.audio_fc = nn.Linear(audio_dim, embed_dim)
        self.video_fc = nn.Linear(video_dim, embed_dim)

    def forward(self, encoded_audio, encoded_video):
        """
        Args:
            encoded_audio: Tensor of shape (batch_size, seq_len, audio_dim)
            encoded_video: Tensor of shape (batch_size, seq_len, video_dim)
        Returns:
            Combined tensor of shape (batch_size, total_seq_len, embed_dim)
        """
        audio_embed = self.audio_fc(encoded_audio)  # (batch_size, seq_len, embed_dim)
        video_embed = self.video_fc(encoded_video)  # (batch_size, seq_len, embed_dim)
        combined = torch.cat((audio_embed, video_embed), dim=1)  # (batch_size, 2*seq_len, embed_dim)
        return combined

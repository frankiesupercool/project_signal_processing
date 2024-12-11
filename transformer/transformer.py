import torch.nn as nn
from positional_encoding import PositionalEncoding
from modality_encoder import ModalityEncoder

class TransformerModel(nn.Module):
    """
    Transformer model that encodes modalities and decodes to clean audio.
    """

    def __init__(self, audio_dim, video_dim, embed_dim=768, nhead=8, num_layers=3,
                 dim_feedforward=532, max_seq_length=1024, denoiser_decoder=None):
        super(TransformerModel, self).__init__()
        self.modality_encoder = ModalityEncoder(audio_dim, video_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Integration of the denoiser's decoder
        if denoiser_decoder is not None:
            self.denoiser_decoder = denoiser_decoder  # Pretrained denoiser's decoder
        else:
            # Define a simple upsampling decoder if no denoiser decoder is provided
            self.denoiser_decoder = nn.Sequential(
                nn.Linear(embed_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 64000)  # Map to waveform length
            )

    def forward(self, encoded_audio, encoded_video):
        """
        Args:
            encoded_audio: Tensor of shape (batch_size, audio_seq_len, audio_dim)
            encoded_video: Tensor of shape (batch_size, video_seq_len, video_dim)
        Returns:
            clean_audio: Tensor of shape (batch_size, clean_audio_length)
        """
        # Encode modalities
        combined = self.modality_encoder(encoded_audio, encoded_video)  # (batch_size, total_seq_len, embed_dim)
        combined = self.positional_encoding(combined)  # Add positional encoding

        # Transformer expects input as (seq_len, batch_size, embed_dim)
        transformer_input = combined.transpose(0, 1)  # (total_seq_len, batch_size, embed_dim)
        transformer_output = self.transformer_encoder(transformer_input)  # (total_seq_len, batch_size, embed_dim)
        transformer_output = transformer_output.transpose(0, 1)  # (batch_size, total_seq_len, embed_dim)

        # Decode to clean audio
        clean_audio = self.denoiser_decoder(transformer_output)  # (batch_size, total_seq_len, 64000)

        # Flatten the last two dimensions if necessary
        clean_audio = clean_audio.view(clean_audio.size(0), -1)  # (batch_size, 64000)

        return clean_audio

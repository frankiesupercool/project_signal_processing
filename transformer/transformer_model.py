import torch
import torch.nn as nn

from config import batch_size
from transformer.modality_encoder import ModalityEncoder
from transformer.positional_encoder import PositionalEncoder

class TransformerModel(nn.Module):
    """
    Transformer model that encodes modalities and decodes to clean audio.
    """

    def __init__(self, audio_dim, video_dim, embed_dim=768, nhead=8, num_layers=3,
                 dim_feedforward=532, max_seq_length=1024, denoiser_decoder=None):
        super(TransformerModel, self).__init__()

        self.audio_proj = nn.Linear(audio_dim, embed_dim)  # Project audio to embed_dim
        self.video_proj = nn.Linear(video_dim, embed_dim)

        self.positional_encoder = PositionalEncoder(d_model=embed_dim, max_len=max_seq_length, zero_pad=False, scale=True)

        self.audio_modality_encoder = ModalityEncoder(embed_dim=embed_dim)
        self.video_modality_encoder = ModalityEncoder(embed_dim=embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
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
        # use projection for same dimension, otherwise adding data + positional + modality_enc would not work
        audio_proj = self.audio_proj(encoded_audio)  # (batch_size, audio_seq_len, embed_dim)
        video_proj = self.video_proj(encoded_video)  # (batch_size, video_seq_len, embed_dim)

        # sinusoidal positional encoding
        positional_audio_encoding = self.positional_encoder(audio_proj)
        positional_video_encoding = self.positional_encoder(video_proj)

        # modality encoding
        modality_audio_encoded = self.audio_modality_encoder(audio_proj)  # (batch_size, total_seq_len, embed_dim)
        modality_video_encoded = self.video_modality_encoder(video_proj)  # (batch_size, total_seq_len, embed_dim)

        # A + PE + ME
        audio_input = audio_proj + positional_audio_encoding + modality_audio_encoded
        # V + PE + ME
        video_input = video_proj + positional_video_encoding + modality_video_encoded

        # concatenate along the temporal dimension
        combined = torch.cat([audio_input, video_input], dim=1)

        # Transformer expects input as (seq_len, batch_size, embed_dim)
        transformer_input = combined.transpose(0, 1)  # (total_seq_len, batch_size, embed_dim)
        transformer_output = self.transformer_encoder(transformer_input)  # (total_seq_len, batch_size, embed_dim)
        transformer_output = transformer_output.transpose(0, 1)  # (batch_size, total_seq_len, embed_dim)

        aggregated_output = torch.mean(transformer_output, dim=1)  # (batch_size, embed_dim)
        # Decode to clean audio
        clean_audio = self.denoiser_decoder(aggregated_output)  # (batch_size, total_seq_len, 64000)


        return clean_audio

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
        if encoded_video.dim() == 4 and encoded_video.size(1) == 1:
            encoded_video = encoded_video.squeeze(1)  # (batch_size, video_seq_len, video_dim)
            print(f"encoded_video shape after squeeze: {encoded_video.shape}")
        elif encoded_video.dim() == 4 and encoded_video.size(1) > 1:
            # Handle multiple channels if applicable
            encoded_video = encoded_video.mean(dim=1)  # Example: Average over channels
            print(f"encoded_video shape after averaging channels: {encoded_video.shape}")
        # use projection for same dimension, otherwise adding data + positional + modality_enc would not work
        audio_proj = self.audio_proj(encoded_audio)  # (batch_size, audio_seq_len, embed_dim)
        video_proj = self.video_proj(encoded_video)  # (batch_size, video_seq_len, embed_dim)
        print(f"audio_proj shape: {audio_proj.shape}")
        print(f"video_proj shape: {video_proj.shape}")
        # sinusoidal positional encoding
        positional_audio_encoding = self.positional_encoder(audio_proj)
        positional_video_encoding = self.positional_encoder(video_proj)
        print(f"positional_audio_encoding shape: {positional_audio_encoding.shape}")
        print(f"positional_video_encoding shape: {positional_video_encoding.shape}")
        # modality encoding
        modality_audio_encoded = self.audio_modality_encoder(audio_proj)  # (batch_size, total_seq_len, embed_dim)
        modality_video_encoded = self.video_modality_encoder(video_proj)  # (batch_size, total_seq_len, embed_dim)
        print(f"modality_audio_encoded shape: {modality_audio_encoded.shape}")
        print(f"modality_video_encoded shape: {modality_video_encoded.shape}")
        # A + PE + ME
        audio_input = audio_proj + positional_audio_encoding + modality_audio_encoded
        # V + PE + ME
        video_input = video_proj + positional_video_encoding + modality_video_encoded
        print(f"audio_input shape: {audio_input.shape}")
        print(f"video_input shape: {video_input.shape}")
        # concatenate along the temporal dimension
        combined = torch.cat([audio_input, video_input], dim=1)
        print(f"combined shape: {combined.shape}")
        # Transformer expects input as (seq_len, batch_size, embed_dim)
        transformer_input = combined.transpose(0, 1)  # (total_seq_len, batch_size, embed_dim)
        transformer_output = self.transformer_encoder(transformer_input)  # (total_seq_len, batch_size, embed_dim)
        transformer_output = transformer_output.transpose(0, 1)  # (batch_size, total_seq_len, embed_dim)

        aggregated_output = torch.mean(transformer_output, dim=1)  # (batch_size, embed_dim)
        # Decode to clean audio
        clean_audio = self.denoiser_decoder(aggregated_output)  # (batch_size, total_seq_len, 64000)
        # Check for NaNs or Infs in output
        if not torch.isfinite(clean_audio).all():
            raise ValueError("Model output contains NaNs or Infs")

        return clean_audio

import torch
import torch.nn as nn

from config import batch_size
from transformer.modality_encoder import ModalityEncoder
from transformer.positional_encoder import PositionalEncoder
from denoiser import pretrained
import linecache  # Import linecache for reading specific lines from files
from video_encoding.video_encoder_service import VideoEncodingService
from utils.device_utils import get_device

class TransformerModel(nn.Module):
    """
    Transformer model that encodes modalities and decodes to clean audio.
    """

    def __init__(self, audio_dim, video_dim, densetcn_options,allow_size_mismatch, backbone_type, use_boundary, relu_type, num_classes, model_path, embed_dim=768, nhead=8, num_layers=3,
                 dim_feedforward=532, max_seq_length=1024, denoiser_decoder=None):
        super(TransformerModel, self).__init__()

        # Initialise Video
        self.densetcn_options = densetcn_options
        self.allow_size_mismatch = allow_size_mismatch
        self.backbone_type = backbone_type
        self.use_boundary = use_boundary
        self.relu_type = relu_type
        self.num_classes = num_classes
        self.model_path = model_path
        self.device = get_device()

        self.video_encoding_service = VideoEncodingService(
            allow_size_mismatch,
            model_path,
            use_boundary,
            relu_type,
            num_classes,
            backbone_type,
            densetcn_options)

        # Load the pretrained denoiser model
        self.model = pretrained.dns64()
        # Initialize the denoiser encoder
        self.encoder = self.model.encoder.to(self.device)

        self.audio_proj = nn.Linear(audio_dim, embed_dim).to(self.device)  # Project audio to embed_dim
        self.video_proj = nn.Linear(video_dim, embed_dim).to(self.device)

        self.positional_encoder = PositionalEncoder(d_model=embed_dim, max_len=max_seq_length, zero_pad=False, scale=True).to(self.device)

        self.audio_modality_encoder = ModalityEncoder(embed_dim=embed_dim).to(self.device)
        self.video_modality_encoder = ModalityEncoder(embed_dim=embed_dim).to(self.device)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True).to(self.device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)

        # Integration of the denoiser's decoder
        if denoiser_decoder is not None:
            self.denoiser_decoder = denoiser_decoder.to(device)  # Pretrained denoiser's decoder
        else:
            # Define a simple upsampling decoder if no denoiser decoder is provided
            self.denoiser_decoder = nn.Sequential(
                nn.Linear(embed_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 64000)  # Map to waveform length
            ).to(self.device)

    def _encode_audio(self, audio):
        with torch.no_grad():
            encoded_audio = audio
            for i, layer in enumerate(self.encoder):
                encoded_audio = layer(encoded_audio)

        # permute to [batch_size, seq_len, embed_dim]
        encoded_audio = encoded_audio.permute(0, 2, 1)  # [64, 61, 1024]
        return encoded_audio


    def _encode_video(self, video):
        encoded_video = self.video_encoding_service.generate_encodings(video)
        encoded_video = encoded_video.squeeze(0)
        return encoded_video

    # TODO rename preprocessed_audio and preprocessed_video maybe
    def forward(self, preprocessed_audio, preprocessed_video):
        """
        Args:
            encoded_audio: Tensor of shape (batch_size, audio_seq_len, audio_dim)
            encoded_video: Tensor of shape (batch_size, video_seq_len, video_dim)
        Returns:
            clean_audio: Tensor of shape (batch_size, clean_audio_length)
        """
        preprocessed_audio = preprocessed_audio.to(self.device)
        preprocessed_video = preprocessed_video.to(self.device)

        encoded_audio = self._encode_audio(preprocessed_audio)
        encoded_video = self._encode_video(preprocessed_video)

        if encoded_video.dim() == 4 and encoded_video.size(1) == 1:
            encoded_video = encoded_video.squeeze(1)  # (batch_size, video_seq_len, video_dim)
        elif encoded_video.dim() == 4 and encoded_video.size(1) > 1:
            # Handle multiple channels if applicable
            encoded_video = encoded_video.mean(dim=1)  # Example: Average over channels
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
        # Check for NaNs or Infs in output
        if not torch.isfinite(clean_audio).all():
            raise ValueError("Model output contains NaNs or Infs")

        return clean_audio

import torch
import torch.nn as nn
import torchaudio.functional

import config
from transformer.modality_encoder import ModalityEncoder
from transformer.positional_encoder import PositionalEncoder
from denoiser import pretrained
from video_encoding.video_encoder_service import VideoPreprocessingService

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

        self.lipreading_preprocessing = VideoPreprocessingService(
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
        self.encoder = self.model.encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.audio_proj = nn.Linear(audio_dim, embed_dim)  # Project audio to embed_dim
        self.video_proj = nn.Linear(video_dim, embed_dim)

        self.upsampled_sample_rate = config.upsampled_sample_rate
        self.target_rate = config.sample_rate

        self.positional_encoder = PositionalEncoder(d_model=embed_dim, max_len=max_seq_length, zero_pad=False, scale=True)

        self.audio_modality_encoder = ModalityEncoder(embed_dim=embed_dim)
        self.video_modality_encoder = ModalityEncoder(embed_dim=embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # projection layer
        self.proj_to_decoder = nn.Linear(embed_dim, 1024)
        # TODO change in features so it is a variable
        self.final_projection = nn.Linear(1024, config.fixed_length)

        # Integration of the denoiser's decoder
        if denoiser_decoder is not None:
            self.denoiser_decoder = denoiser_decoder  # Pretrained denoiser's decoder
        else:
            print("Warning: denoiser is None! Fallback to simple upsampling decoder")
            # Define a simple upsampling decoder if no denoiser decoder is provided
            self.denoiser_decoder = nn.Sequential(
                nn.Linear(embed_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, config.fixed_length)  # Map to waveform length
            )

        for param in self.denoiser_decoder.parameters():
            param.requires_grad = False

    def _encode_audio(self, audio):
        """
        Runs the input audio through the encoder, storing skip connections.
        Expects audio of shape [batch, channels, time].
        Returns:
            encoded_audio: [batch, time, features]
            skip_connections: list of tensors, each with shape [batch, time, features]
        """
        skip_connections = []
        x = audio
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)  # Save output of each encoder block
        # Permute final output to [batch, time, features]
        encoded_audio = x.permute(0, 2, 1)
        # Also permute each skip connection for easier fusion later
        skip_connections = [s.permute(0, 2, 1) for s in skip_connections]
        return encoded_audio, skip_connections

    def _decode_audio(self, audio, skip_connections):
        """
        Decodes the fused audio representation, integrating skip connections.
        Args:
            audio: Tensor of shape [batch, seq_len, embed_dim] (transformer output)
            skip_connections: list of tensors from the encoder.
        Returns:
            decoded_audio: Tensor of shape [batch, clean_audio_length]
        """
        # Project transformer output to decoder channels and permute for Conv1d layers
        decoded_audio = self.proj_to_decoder(audio)  # [batch, seq_len, 1024]
        decoded_audio = decoded_audio.permute(0, 2, 1)  # [batch, 1024, seq_len]

        # Convert decoder modules into a list; assume they are iterable
        # (This works for ModuleList or a Sequential module.)
        decoder_layers = list(self.denoiser_decoder.children()) \
                         if isinstance(self.denoiser_decoder, nn.Sequential) \
                         else list(self.denoiser_decoder)

        num_skips = len(skip_connections)
        for i, layer in enumerate(decoder_layers):
            # Add corresponding skip connection (in reverse order)
            if i < num_skips:
                skip = skip_connections[-(i+1)]
                # Align temporal dimensions if needed
                if skip.size(1) > decoded_audio.size(2):
                    skip = skip[:, :decoded_audio.size(2), :]
                elif skip.size(1) < decoded_audio.size(2):
                    decoded_audio = decoded_audio[:, :, :skip.size(1)]
                # Permute skip to match [batch, channels, seq_len]
                skip = skip.permute(0, 2, 1)
                decoded_audio = decoded_audio + skip
            decoded_audio = layer(decoded_audio)

        # If the decoder output does not match the expected fixed length, apply adaptive pooling
        if decoded_audio.size(1) == 1 and decoded_audio.size(2) != config.fixed_length:
            pooled = nn.functional.adaptive_avg_pool1d(decoded_audio, output_size=1024)
            pooled = pooled.squeeze(1)
            decoded_audio = self.final_projection(pooled)

        # Resample the output to the target sample rate
        decoded_audio = torchaudio.functional.resample(
            decoded_audio,
            orig_freq=self.upsampled_sample_rate,
            new_freq=self.target_rate
        )
        return decoded_audio



    def _encode_video(self, video):
        encoded_video = self.lipreading_preprocessing.generate_encodings(video)
        encoded_video = encoded_video.squeeze(0)
        return encoded_video

    def forward(self, preprocessed_audio, preprocessed_video):
        """
        Args:
            encoded_audio: Tensor of shape (batch_size, audio_seq_len, audio_dim)
            encoded_video: Tensor of shape (batch_size, video_seq_len, video_dim)
        Returns:
            clean_audio: Tensor of shape (batch_size, clean_audio_length)
        """
        preprocessed_audio = preprocessed_audio
        preprocessed_video = preprocessed_video

        encoded_audio, skip_connections = self._encode_audio(preprocessed_audio)
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

        #aggregated_output = torch.mean(transformer_output, dim=1)  # (batch_size, embed_dim)
        # Decode to clean audio
        clean_audio = self._decode_audio(transformer_output, skip_connections) # (batch_size, total_seq_len, 64000)
        # Check for NaNs or Infs in output
        if not torch.isfinite(clean_audio).all():
            raise ValueError("Model output contains NaNs or Infs")

        return clean_audio

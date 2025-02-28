import math
import torch as th
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchaudio
from video_encoding.video_encoder_service import VideoPreprocessingService


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, embed_dim, max_len=1024):
        super().__init__()
        self.d_model = embed_dim
        self.max_len = max_len
        pe = th.zeros(max_len, embed_dim)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, embed_dim, 2, dtype=th.float) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # to get shape (1, max_len, embed_dim)

    def forward(self, x):
        # [batch, seq_len, embed_dim]
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


class ModalityEncoder(nn.Module):
    """Adds a learnable modality-specific vector to the input"""

    def __init__(self, embed_dim=768):
        super().__init__()
        self.modality_encoding = nn.Parameter(th.randn(1, 1, embed_dim))

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        return x + self.modality_encoding.expand_as(x)


class InternalAVTransformer(nn.Module):
    """
    Internally used AVTransformer, sets up actual transformer with 3 layers, 8 heads, and model size 532.
    Input projected audio and projected video adds positional and modality encodings to each.
    Concatenates them along the temporal dimension.
    """

    def __init__(self, embed_dim=768, max_seq_length=1024):
        super().__init__()
        nhead, num_layers, dim_feedforward = 8, 3, 532  # transformer setup as described in paper

        self.positional_encoder = PositionalEncoder(embed_dim=embed_dim, max_len=max_seq_length)
        self.audio_modality_encoder = ModalityEncoder(embed_dim=embed_dim)
        self.video_modality_encoder = ModalityEncoder(embed_dim=embed_dim)

        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, audio_tokens, video_tokens):
        """
        Generates positional and modality encoding, concat on temporal dim
        Calls transformer with fused data
        Args:
            audio_tokens: [batch, conv channels, embed_dim]
            video_tokens: [batch, frames, embed_dim]

        Returns: Audio sequence part of transformer output
        """
        # A + PE + MEy
        audio_tokens = audio_tokens + self.positional_encoder(audio_tokens) + self.audio_modality_encoder(audio_tokens)
        # V + PE + ME
        video_tokens = video_tokens + self.positional_encoder(video_tokens) + self.video_modality_encoder(video_tokens)
        fused = th.cat([audio_tokens, video_tokens], dim=1)
        fused_out = self.transformer_encoder(fused)
        audio_seq = audio_tokens.size(1)
        return fused_out[:, :audio_seq, :]


class AVTransformer(nn.Module):
    """
    Set up U-Net denoiser style audio encoder decoder.
    https://github.com/facebookresearch/denoiser/blob/main/denoiser/demucs.py
    Calls preprocessing and projects preprocessed audio (noised audio) and preprocessed video.
    Calls InternalAVTransformer where the actual prediction is done.
    """

    def __init__(self,
                 # video preprocessing params
                 densetcn_options,
                 allow_size_mismatch,
                 backbone_type,
                 use_boundary,
                 relu_type,
                 num_classes,
                 model_path,
                 # U-Net denoiser params
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 padding=2,
                 growth=2,
                 max_hidden=10000,
                 # additional params
                 video_preprocessing_dim=512,
                 embed_dim=768,  # projection dimension for transformer
                 max_seq_length=1024,
                 orig_sample_rate=16000,
                 upsampled_sample_rate=51200):
        super().__init__()

        # Initialise video preprocessing
        self.lipreading_preprocessing = VideoPreprocessingService(
            allow_size_mismatch=allow_size_mismatch,
            model_path=model_path,
            use_boundary=use_boundary,
            relu_type=relu_type,
            num_classes=num_classes,
            backbone_type=backbone_type,
            densetcn_options=densetcn_options)

        # U-Net denoiser setup
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1)
        ch_scale = 2

        # Iterative U-Net denoiser encoder layer setup
        for index in range(depth):
            encode = [
                nn.Conv1d(chin, hidden, kernel_size, stride, padding),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1),
                activation,
            ]
            self.encoder.append(nn.Sequential(*encode))
            decode = [
                nn.Conv1d(hidden, ch_scale * hidden, 1),
                activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride, padding),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        # Resampling layers
        self.upsample = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=upsampled_sample_rate)
        self.downsample = torchaudio.transforms.Resample(orig_freq=upsampled_sample_rate, new_freq=orig_sample_rate)

        # Audio projection layer - project last encoder layer to embedding dim
        self.audio_proj = nn.Linear(chin, embed_dim)
        # Video projection layer - project video preprocessing to embedding dim
        self.video_proj = nn.Linear(video_preprocessing_dim, embed_dim)

        # Internal actual AV transformer
        self.av_transformer = InternalAVTransformer(embed_dim=embed_dim, max_seq_length=max_seq_length)

    def _encode_audio(self, audio):
        """
        Encode up-sampled audio waveform, seq_length_enc = 320 by conv
        Args:
            audio: Up-sampled audio waveform - [batch, channels, seq_length]

        Returns:
            x: Encoded audio - [batch, channel dim (embed dim), seq_length_enc]
            skip_connections: Skip connections for decoder
        """
        skip_connections = []
        x = audio
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
        return x, skip_connections

    def _decode_audio(self, x, skip_connections):
        """
        Decoder for output of InternalAVTransformer
        Args:
            x: Transformer output to decode - [batch, embed_dim, seq_length_enc]
            skip_connections: Skip connection from encoder

        Returns: Decoded audio waveform [batch, 1, seq_length (time upsampled)]
        """
        for decode in self.decoder:
            skip = skip_connections.pop(-1)
            # Align temporal dimensions
            if skip.size(-1) < x.size(-1):
                x = x[..., :skip.size(-1)]
            else:
                skip = skip[..., :x.size(-1)]
            x = x + skip
            x = decode(x)
        return x

    def forward(self, audio, video):
        """
        Upsamples audio to match denoiser encoder, projects audio and video embedding space. Encodes preprocessed audio
        with U-Net denoiser encoder, preprocesses video. Feeds both into InternalAVTransformer. Decodes transformer
        audio output with the denoiser decoder.

        Args:
            audio: Noisy audio input waveform [batch, 1, seq_length] (already preprocessed by dataset)
            video: Video input [batch, frames, 96, 96] (96x96 crop)

        Returns:
            clean_audio: Predicted audio waveform - [batch, seq_length]
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Upsample audio for conv encoder
        audio = self.upsample(audio)  # [batch, 1, time_up]
        # Encode audio
        audio_enc, skip_connections = self._encode_audio(audio)
        # Encoder output shape [batch, embed_dim, seq_length_encoded] permute to [batch, seq, embed_dim]
        audio_tokens = audio_enc.permute(0, 2, 1)
        audio_tokens = self.audio_proj(audio_tokens)

        # Preprocess video
        video_encoded = self.lipreading_preprocessing.generate_encodings(video)
        video_encoded = video_encoded.float()
        video_tokens = self.video_proj(video_encoded)

        # Call to InternalAVTransformer for prediction
        predicted_audio_tokens = self.av_transformer(audio_tokens, video_tokens)

        # Back-projection of transformer output to match decoder with linear layer (1x1 conv)
        proj_back = (nn.Conv1d(predicted_audio_tokens.size(-1), audio_enc.size(1), kernel_size=1)
                     .to(predicted_audio_tokens.device))
        # [batch, seq, embed_dim] permute to decoder input shape [batch, embed_dim, seq]
        predicted_audio_tokens = predicted_audio_tokens.transpose(1, 2)
        conv_in = proj_back(predicted_audio_tokens)

        decoded_audio = self._decode_audio(conv_in, skip_connections)  # [batch, 1, time_up]
        # Downsample to original sample rate
        decoded_audio = self.downsample(decoded_audio)
        return decoded_audio

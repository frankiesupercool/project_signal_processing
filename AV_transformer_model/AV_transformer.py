import math
import torch as th
from torch import nn
import torchaudio
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from video_encoding.video_encoder_service import VideoPreprocessingService

# --- Additional modules for modality and positional encoding ---

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoder."""
    def __init__(self, d_model, max_len=1024, zero_pad=False, scale=False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2, dtype=th.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        if zero_pad:
            pe[0, :] = 0
        self.register_buffer('pe', pe.unsqueeze(0))  # shape (1, max_len, d_model)
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]

class ModalityEncoder(nn.Module):
    """Adds a learnable modality-specific vector to the input."""
    def __init__(self, embed_dim=768):
        super().__init__()
        self.modality_encoding = nn.Parameter(th.randn(1, 1, embed_dim))
    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        return x + self.modality_encoding.expand_as(x)

# --- Modified TransformerModel that fuses audio and video modalities ---

class TransformerFusion(nn.Module):
    """
    This module takes in projected audio and video tokens,
    adds positional and modality encodings to each,
    concatenates them along the temporal dimension,
    and fuses them via a transformer encoder.
    """
    def __init__(self, embed_dim=768, nhead=8, num_layers=3, dim_feedforward=532, max_seq_length=1024):
        super().__init__()
        self.positional_encoder = PositionalEncoder(d_model=embed_dim, max_len=max_seq_length)
        self.audio_modality_encoder = ModalityEncoder(embed_dim=embed_dim)
        self.video_modality_encoder = ModalityEncoder(embed_dim=embed_dim)
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, audio_tokens, video_tokens):
        # audio_tokens: [B, T_a, embed_dim]
        # video_tokens: [B, T_v, embed_dim]
        audio_tokens = audio_tokens + self.positional_encoder(audio_tokens) + self.audio_modality_encoder(audio_tokens)
        video_tokens = video_tokens + self.positional_encoder(video_tokens) + self.video_modality_encoder(video_tokens)
        # Concatenate along temporal dimension (assume audio tokens come first)
        fused = th.cat([audio_tokens, video_tokens], dim=1)  # [B, T_a+T_v, embed_dim]
        fused = self.transformer_encoder(fused)  # [B, T_a+T_v, embed_dim]
        # Return only the first T_a tokens (the audio tokens)
        T_a = audio_tokens.size(1)
        return fused[:, :T_a, :]

# --- Modified VoiceFormerAVE with video encoding branch ---

class AVTransformer(nn.Module):
    def __init__(self,
                 densetcn_options,
                 allow_size_mismatch,
                 backbone_type,
                 use_boundary,
                 relu_type,
                 num_classes,
                 model_path,
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 padding=2,
                 resample=3.2,
                 growth=2,
                 max_hidden=10_000,
                 normalize=False,  # we assume dataset normalization already
                 glu=True,
                 floor=1e-3,
                 video_chin=512,
                 d_hid=532,
                 num_encoder_layers=3,   # for the transformer part (in the conv branch)
                 num_heads=8,
                 embed_dim=768,          # projection dimension for transformer
                 transformer_layers=3,
                 transformer_ff_dim=532,
                 max_seq_length=1024):
        super().__init__()

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

        # Save parameters for conv encoder/decoder as in original VoiceFormerAVE:
        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.floor = floor
        self.resample = resample
        self.normalize = normalize  # if True, model normalizes, but dataset already does
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        self.video_chin = video_chin
        # These layers remain for conv encoding:
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
        # Resampling layers:
        self.upsample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=51200)
        self.downsample = torchaudio.transforms.Resample(orig_freq=51200, new_freq=16000)

        # --- Projection layers to create transformer tokens ---
        # For audio: after conv encoder, output shape: [B, C, T] then permuted to [B, T, C]
        self.audio_proj = nn.Linear(chin, embed_dim)
        # For video: assume video preprocessing produces features with dimension `video_chin`
        self.video_proj = nn.Linear(video_chin, embed_dim)

        # The transformer fusion module (as defined above)
        self.av_transformer = TransformerFusion(embed_dim=embed_dim,
                                                 nhead=num_heads,
                                                 num_layers=transformer_layers,
                                                 dim_feedforward=transformer_ff_dim,
                                                 max_seq_length=max_seq_length)

    def _encode_audio(self, audio):
        # audio: [B, channels, time]
        skip_connections = []
        x = audio
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
        return x, skip_connections  # x shape: [B, C, T]

    def _decode_audio(self, x, skip_connections):
        # x: [B, C, T]
        for decode in self.decoder:
            skip = skip_connections.pop(-1)
            # If necessary, align temporal dimensions:
            if skip.size(-1) < x.size(-1):
                x = x[..., :skip.size(-1)]
            else:
                skip = skip[..., :x.size(-1)]
            x = x + skip
            x = decode(x)
        return x

    def forward(self, audio, visual, speaker_embedding=None, use_video_backbone=True):
        """
        Args:
            audio: input waveform [B, 1, time] (already preprocessed by dataset)
            visual: preprocessed video features [B, video_seq, video_chin]
        Returns:
            clean_audio: waveform [B, time]
        """
        # Bypass normalization in model if dataset normalization is used.
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if self.normalize:
            mono = audio.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            audio = audio / (self.floor + std)
        else:
            std = 1

        length = audio.shape[-1]
        # Upsample audio for conv encoder
        x = self.upsample(audio)  # [B, 1, time_up]
        # Pass through conv encoder to obtain features and skip connections:
        conv_out, skip_connections = self._encode_audio(x)  # conv_out: [B, C, T_conv]
        # Prepare audio tokens: permute to [B, T_conv, C] then project:
        audio_tokens = conv_out.permute(0, 2, 1)  # [B, T_conv, C]
        audio_tokens = self.audio_proj(audio_tokens)  # [B, T_conv, embed_dim]

        # For video, if you want to use the full encoding service:
        if use_video_backbone:
            # Assume visual is a raw video file path or less processed video data.
            video_encoded = self.lipreading_preprocessing.generate_encodings(visual)
            video_encoded = video_encoded.float()
        else:
            # Otherwise, assume visual is already preprocessed.
            video_encoded = visual

        # Process video branch:
        video_tokens = self.video_proj(video_encoded)  # [B, video_seq, embed_dim]

        # Fuse audio and video tokens via transformer:
        fused_audio_tokens = self.av_transformer(audio_tokens, video_tokens)  # [B, T_conv, embed_dim]

        # (Optionally, you could project fused tokens back to conv channel dimension.)
        # For simplicity, we use a linear layer (1x1 conv) here:
        proj_back = nn.Conv1d(fused_audio_tokens.size(-1), conv_out.size(1), kernel_size=1).to(fused_audio_tokens.device)
        # Permute to [B, embed_dim, T_conv]
        fused_back = fused_audio_tokens.transpose(1, 2)
        conv_in = proj_back(fused_back)  # [B, C, T_conv]

        # Decode audio features using conv decoder and skip connections:
        decoded = self._decode_audio(conv_in, skip_connections)  # [B, 1, time_up_decoded]
        # Downsample back to target sample rate:
        decoded = self.downsample(decoded)
        decoded = decoded[..., :length]
        return decoded * std
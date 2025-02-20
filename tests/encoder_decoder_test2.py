import torch
import torchaudio
import torch.nn as nn
from denoiser import pretrained
import config

# Load the pretrained denoiser model
denoiser = pretrained.dns64()
encoder = denoiser.encoder
decoder = denoiser.decoder

# Set to eval mode (disable dropout, batchnorm updates, etc.)
encoder.eval()
decoder.eval()


def encode_audio(audio):
    """
    Runs input audio through the encoder, storing skip connections.
    """
    skip_connections = []
    x = audio
    for layer in encoder:
        x = layer(x)
        skip_connections.append(x)  # Save output of each encoder block

    # Permute final output to [batch, time, features]
    encoded_audio = x.permute(0, 2, 1)
    skip_connections = [s.permute(0, 2, 1) for s in skip_connections]  # Adjust shape

    return encoded_audio, skip_connections


def decode_audio(encoded_audio, skip_connections):
    """
    Decodes audio, integrating skip connections.
    """
    decoded_audio = encoded_audio.permute(0, 2, 1)  # [batch, features, time]

    decoder_layers = list(decoder.children()) if isinstance(decoder, nn.Sequential) else list(decoder)

    num_skips = len(skip_connections)
    for i, layer in enumerate(decoder_layers):
        # Add skip connection (reverse order)
        if i < num_skips:
            skip = skip_connections[-(i + 1)]
            # Align dimensions if needed
            if skip.size(1) > decoded_audio.size(2):
                skip = skip[:, :decoded_audio.size(2), :]
            elif skip.size(1) < decoded_audio.size(2):
                decoded_audio = decoded_audio[:, :, :skip.size(1)]
            skip = skip.permute(0, 2, 1)  # [batch, channels, time]
            decoded_audio = decoded_audio + skip  # Merge skip connection
        decoded_audio = layer(decoded_audio)

    return decoded_audio

# Step 1: Load your existing audio file (ensure it is a mono channel .wav file)
audio_path = '../' + config.PRETRAIN_DATA_PATH + '/U4Lx5rcii38/00001.wav'  # Replace with your file path

audio_waveform, orig_sample_rate = torchaudio.load(audio_path)

# Ensure the audio is mono (1 channel), and resample if necessary
if audio_waveform.shape[0] > 1:
    audio_waveform = audio_waveform.mean(dim=0, keepdim=True)  # Convert to mono if stereo

# Resample audio if necessary to match the sample rate the model expects (e.g., 16000 Hz)
target_sample_rate = config.sample_rate  # Ensure your config contains the correct sample rate (e.g., 16000)
if orig_sample_rate != target_sample_rate:
    audio_waveform = torchaudio.functional.resample(audio_waveform, orig_sample_rate, target_sample_rate)

# Ensure audio has the shape [batch_size, channels, time] for model processing
audio_waveform = audio_waveform.unsqueeze(0)  # Add batch dimension (1, 1, time)

# Step 2: Encode the audio
with torch.no_grad():
    encoded_audio, skip_connections = encode_audio(audio_waveform)
    print(f"Encoded Audio Shape: {encoded_audio.shape}")
    print(f"Number of Skip Connections: {len(skip_connections)}")

# Step 3: Decode the audio
with torch.no_grad():
    decoded_audio = decode_audio(encoded_audio, skip_connections)
    print(f"Decoded Audio Shape: {decoded_audio.shape}")

    # Save the tensor as a WAV file
    output_file = "output_audio.wav"
    torchaudio.save(output_file, decoded_audio.squeeze(0), sample_rate=target_sample_rate)

    print(f"Audio saved to {output_file}")

# Step 4: Verify output
if torch.isnan(decoded_audio).any() or torch.isinf(decoded_audio).any():
    print("Warning: NaNs or Infs detected in decoded audio output!")
else:
    print("Success! Encoder and decoder are functioning correctly.")

from IPython import display as disp
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
from utils.device_utils import get_device, set_seed

device = get_device()

model = pretrained.dns64()
encoder = model.encoder.to(device)
decoder = model.decoder


wav, sr = torchaudio.load('../data/pretrain/aySy3tSYGYw/00001.wav')
wav = convert_audio(wav, sr, model.sample_rate, model.chin)
wav = wav.to(device)

if wav.dim() == 2:
    # Assuming wav shape is [channels, samples], add batch dimension
    wav = wav.unsqueeze(0)  # Now wav has shape [1, channels, samples]

# Pass the wav through each layer of the encoder sequentially
encoded_wav = wav
for layer in encoder:
    encoded_wav = layer(encoded_wav)

# encoded_wav now contains the encoded representation of your audio








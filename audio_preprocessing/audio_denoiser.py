from IPython import display as disp
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
from utils.device_utils import get_device, set_seed

device = get_device()

model = pretrained.dns64()
encoder = model.encoder
decoder = model.decoder








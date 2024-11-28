import os
import torch
from torch.utils.data import Dataset
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
from noise import Remix, RevEcho, BandMask, Shift
from IPython import display as disp
from utils.device_utils import get_device, set_seed

"""
Preprocessing the audio using the encoder of the denoiser model in 
https://github.com/facebookresearch/denoiser


"""

class AudioPreprocessDataset(Dataset):
    def __init__(self, root_dir, split='pretrain', transform=None, sample_rate=16000, cache_dir=None, noise_prob=0.5, noise_dir=None):
        """
        Args:
            root_dir (str): Root directory containing 'pretrain', 'trainval', 'test' folders.
            split (str): One of 'pretrain', 'trainval', 'test' indicating the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
            sample_rate (int): Desired sample rate for audio files.
        """
        self.split = split
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.sample_rate = sample_rate
        self.cache_dir = cache_dir
        self.noise_prob = noise_prob
        self.noise_dir = noise_dir

        self.file_list = self._gather_wav_files()
        self.model = pretrained.dns64()
        # Initialise the denoiser encoder
        self.encoder = self.model.encoder

    def _gather_wav_files(self):
        wav_files = []
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(subdir, file))
        return wav_files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        wav_path = self.file_list[idx]
        # Load audio
        waveform, sr = torchaudio.load(wav_path)
        waveform = convert_audio(waveform, sr, self.model.sample_rate, self.model.chin)

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # Now wav has shape [1, channels, samples]

        # Add noise
        noisy_waveform = add_noise(waveform)

        # Encode using denoiser's encoder
        with torch.no_grad():
            encoded_audio = noisy_waveform
            for layer in self.encoder:
                encoded_audio = layer(encoded_audio)

        sample = {
            'encoded_audio': encoded_audio,
            'clean_waveform': waveform.squeeze(0),
            'noisy_waveform': noisy_waveform.squeeze(0),
            'file_path': wav_path
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
#import pytorch_lightning as pl
import numpy as np


class LRS3WithNoiseDataset(Dataset):
    def __init__(self, lrs3_root, dns_root, snr_db=0, transform=None, sample_rate=16000,
                 mode_prob={'speaker': 0.5, 'noise': 0.5}):
        """
        Args:
            lrs3_root (str): Path to the LRS3 dataset root directory.
            dns_root (str): Path to the DNS dataset root directory.
            snr_db (float): Desired Signal-to-Noise Ratio in decibels.
            transform (callable, optional): Optional transform to be applied on a sample.
            sample_rate (int): Desired sample rate for audio files.
            mode_prob (dict): Probability distribution for selecting mode. Keys should be 'speaker' and 'noise'.
        """
        self.lrs3_root = lrs3_root
        self.dns_root = dns_root
        self.snr_db = snr_db
        self.transform = transform
        self.sample_rate = sample_rate
        self.mode_prob = mode_prob



        self.lrs3_files = self._get_file_list(self.lrs3_root, file_extension='.wav')
        self.dns_files = self._get_file_list(self.dns_root, file_extension='.wav')

        # Build speaker-to-file mapping for speaker separation
        self.speaker_to_files = self._build_speaker_dict(self.lrs3_files)
        self.speakers = list(self.speaker_to_files.keys())

        if len(self.speakers) < 2:
            raise ValueError("Need at least two speakers in LRS3 dataset for speaker separation.")

    def _get_file_list(self, root_dir, file_extension='.wav'):
        """
        Recursively collects all files with the given extension in the root directory.
        """
        file_list = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(file_extension):
                    file_list.append(os.path.join(root, file))
        return file_list

    def _build_speaker_dict(self, file_list):
        """
        Builds a dictionary mapping speaker IDs to their corresponding audio files.
        Assumes that the speaker ID is part of the file path.
        Adjust the `_extract_speaker_id` method if your dataset's structure differs.
        """
        speaker_dict = {}
        for file_path in file_list:
            speaker_id = self._extract_speaker_id(file_path)
            if speaker_id not in speaker_dict:
                speaker_dict[speaker_id] = []
            speaker_dict[speaker_id].append(file_path)
        return speaker_dict

    def _extract_speaker_id(self, file_path):
        """
        Extracts the speaker ID from the file path.
        Modify this method based on your dataset's directory structure.
        Example assumption: '.../LRS3/speaker_id/filename.wav'
        """
        return os.path.basename(os.path.dirname(file_path))

    def __len__(self):
        return len(self.lrs3_files)

    def add_noise_with_snr(self, speech, interference, snr_db):
        """
        Mixes speech with interference (noise or another speaker) at the specified SNR.

        Args:
            speech (torch.Tensor): Clean speech waveform.
            interference (torch.Tensor): Interfering waveform (noise or another speaker's speech).
            snr_db (float): Desired Signal-to-Noise Ratio in decibels.

        Returns:
            torch.Tensor: Mixed waveform.
        """
        # Ensure the waveforms are on the same device
        device = speech.device

        # Calculate power of speech and interference
        speech_power = speech.norm(p=2)
        interference_power = interference.norm(p=2)

        if interference_power == 0:
            raise ValueError("Interference power is zero, cannot compute scaling factor.")

        # Calculate the scaling factor for interference to achieve desired SNR
        snr_linear = 10 ** (snr_db / 20)
        scaling_factor = speech_power / (interference_power * snr_linear)

        # Scale interference and mix
        interference_scaled = interference * scaling_factor
        mixed = speech + interference_scaled

        return mixed

    def __getitem__(self, idx):
        # Load clean speech from LRS3
        lrs3_file = self.lrs3_files[idx]
        speech_waveform, orig_sample_rate = torchaudio.load(lrs3_file)
        speech_waveform = torchaudio.functional.resample(speech_waveform, orig_freq=orig_sample_rate,
                                                         new_freq=self.sample_rate)
        speech_waveform = speech_waveform.squeeze(0)  # Assuming mono; adjust if stereo

        # Decide randomly whether to add another speaker or noise
        mode = random.choices(['speaker', 'noise'], weights=[self.mode_prob.get('speaker', 0.5),
                                                             self.mode_prob.get('noise', 0.5)])[0]

        if mode == 'speaker':
            # Speaker Separation: Add speech from another speaker
            clean_speaker_id = self._extract_speaker_id(lrs3_file)
            # Ensure there's at least one other speaker
            if len(self.speakers) < 2:
                raise ValueError("Not enough speakers for speaker separation.")

            # Select a different speaker
            other_speakers = [spk for spk in self.speakers if spk != clean_speaker_id]
            interfering_speaker_id = random.choice(other_speakers)
            # Select a random file from the interfering speaker
            interfering_file = random.choice(self.speaker_to_files[interfering_speaker_id])
            interfering_waveform, orig_sample_rate = torchaudio.load(interfering_file)
            interfering_waveform = torchaudio.functional.resample(interfering_waveform, orig_freq=orig_sample_rate,
                                                                  new_freq=self.sample_rate)
            interfering_waveform = interfering_waveform.squeeze(0)  # Assuming mono; adjust if stereo

            interference_type = 'speaker'

        elif mode == 'noise':
            # Speech Enhancement: Add background noise from DNS
            dns_file = random.choice(self.dns_files)
            interfering_waveform, orig_sample_rate = torchaudio.load(dns_file)
            interfering_waveform = torchaudio.functional.resample(interfering_waveform, orig_freq=orig_sample_rate,
                                                                  new_freq=self.sample_rate)
            interfering_waveform = interfering_waveform.squeeze(0)  # Assuming mono; adjust if stereo

            interference_type = 'noise'

        else:
            raise ValueError("Invalid mode selected.")

        # Ensure interfering waveform matches speech length
        speech_length = speech_waveform.shape[0]
        interference_length = interfering_waveform.shape[0]

        if interference_length > speech_length:
            # Randomly select a segment
            max_start = interference_length - speech_length
            start_idx = random.randint(0, max_start)
            interfering_waveform = interfering_waveform[start_idx:start_idx + speech_length]
        else:
            # Pad interfering waveform
            padding = speech_length - interference_length
            interfering_waveform = torch.nn.functional.pad(interfering_waveform, (0, padding))

        # Mix speech and interference at desired SNR
        mixture = self.add_noise_with_snr(speech_waveform, interfering_waveform, self.snr_db)

        # Apply any transformations
        if self.transform:
            mixture = self.transform(mixture)

        sample = {
            'mixture': mixture.unsqueeze(0),  # Add channel dimension
            'clean_speech': speech_waveform.unsqueeze(0),
            'interference': interfering_waveform.unsqueeze(0),
            'interference_type': interference_type,
            'file_path': lrs3_file
        }

        return sample

dataset = LRS3WithNoiseDataset(lrs3_root='../data/pretrain', dns_root='../noise_data_set/noise', snr_db=30)

# Access the first sample
sample = dataset[1]

# Print the keys of the sample dictionary
print("Sample Keys:", sample.keys())

# Inspect individual components
print("Mixture Shape:", sample['mixture'].shape)
print("Clean Speech Shape:", sample['clean_speech'].shape)
print("Interference Shape:", sample['interference'].shape)
print("Interference Type:", sample['interference_type'])
print("SNR (dB):", sample.get('snr_db', 'N/A'))  # 'snr_db' may not exist in fixed SNR setup
print("File Path:", sample['file_path'])

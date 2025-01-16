import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from denoiser import pretrained
import linecache  # Import linecache for reading specific lines from files
from video_encoding.video_encoder_service import VideoPreprocessingService
from video_preprocessing.video_preprocessor_simple import VideoPreprocessorSimple


class PreprocessingDataset(Dataset):
    def __init__(self, lrs3_root, dns_root, densetcn_options,allow_size_mismatch, backbone_type, use_boundary, relu_type, num_classes, model_path, snr_db=0, transform=None, sample_rate=16000,
                 mode_prob={'speaker': 0.5, 'noise': 0.5}, fixed_length=64000):

        # todo fixed_frames ?
        """
        Args:
            lrs3_root (str): Path to the LRS3 dataset root directory.
            dns_root (str): Path to the DNS dataset root directory.
            snr_db (float): Desired Signal-to-Noise Ratio in decibels.
            transform (callable, optional): Optional transform to be applied on a sample.
            sample_rate (int): Desired sample rate for audio files.
            mode_prob (dict): Probability distribution for selecting mode. Keys should be 'speaker' and 'noise'.
            fixed_length (int): Fixed length in samples for audio waveforms.
            fixed_frames (int): Fixed number of frames for video sequences.
        """
        self.lrs3_root = lrs3_root
        self.dns_root = dns_root
        self.snr_db = snr_db
        self.transform = transform
        self.sample_rate = sample_rate
        self.mode_prob = mode_prob
        self.fixed_length = fixed_length  # Fixed length for audio samples

        # Initialise Video
        self.densetcn_options = densetcn_options
        self.allow_size_mismatch = allow_size_mismatch
        self.backbone_type = backbone_type
        self.use_boundary = use_boundary
        self.relu_type = relu_type
        self.num_classes = num_classes
        self.model_path = model_path

        # Load the pretrained denoiser model
        self.model = pretrained.dns64()
        # Initialize the denoiser encoder
        self.encoder = self.model.encoder

        # Save paired file paths to a single text file
        self.paired_files_list = 'paired_files.txt'
        self.dns_files_list = 'dns_files.txt'

        if not os.path.exists(self.paired_files_list):
            self._write_paired_file_list(self.lrs3_root, self.paired_files_list, audio_ext='.wav', video_ext='.mp4')
        if not os.path.exists(self.dns_files_list):
            self._write_file_list(self.dns_root, self.dns_files_list, file_extension='.wav')

        self.paired_files_len = sum(1 for _ in open(self.paired_files_list))
        self.dns_files_len = sum(1 for _ in open(self.dns_files_list))

        # Get list of speakers
        self.speakers = self._get_speakers()

        if len(self.speakers) < 2:
            raise ValueError("Need at least two speakers in LRS3 dataset for speaker separation.")

        self.lipreading_preprocessing = VideoPreprocessingService(
            allow_size_mismatch,
            model_path,
            use_boundary,
            relu_type,
            num_classes,
            backbone_type,
            densetcn_options)
        self.video_processor = VideoPreprocessorSimple(target_frames=100, fps=25.0)

    def _write_paired_file_list(self, root_dir, output_file, audio_ext='.wav', video_ext='.mp4'):
        with open(output_file, 'w') as f:
            for speaker in os.listdir(root_dir):
                speaker_dir = os.path.join(root_dir, speaker)
                if not os.path.isdir(speaker_dir):
                    continue
                audio_files = sorted([f for f in os.listdir(speaker_dir) if f.lower().endswith(audio_ext)])
                video_files = sorted([f for f in os.listdir(speaker_dir) if f.lower().endswith(video_ext)])

                # Ensure that both audio and video files exist
                if not audio_files or not video_files:
                    print(f"Warning: Speaker {speaker} does not have both audio and video files.")
                    continue

                # Pair files up to the minimum length
                paired = zip(audio_files, video_files)
                for audio, video in paired:
                    audio_path = os.path.join(speaker_dir, audio)
                    video_path = os.path.join(speaker_dir, video)
                    if os.path.exists(video_path):
                        f.write(f"{audio_path}\t{video_path}\n")
                    else:
                        print(f"Warning: Video file {video_path} does not exist for audio file {audio_path}")

    def _write_file_list(self, root_dir, output_file, file_extension='.wav'):
        with open(output_file, 'w') as f:
            for root, _, files in os.walk(root_dir):
                for file in sorted(files):
                    if file.lower().endswith(file_extension):
                        f.write(os.path.join(root, file) + '\n')

    def _get_speakers(self):
        """
        Returns a list of speaker IDs based on the directory structure.
        Only includes speakers that have at least one .wav file.
        """
        speakers = []
        for d in os.listdir(self.lrs3_root):
            speaker_dir = os.path.join(self.lrs3_root, d)
            if os.path.isdir(speaker_dir):
                wav_files = [f for f in os.listdir(speaker_dir) if f.lower().endswith('.wav')]
                if wav_files:
                    speakers.append(d)
        return speakers

    def _extract_speaker_id(self, file_path):
        """
        Extracts the speaker ID from the file path.
        """
        return os.path.basename(os.path.dirname(file_path))

    def __len__(self):
        return self.paired_files_len

    def pad_or_truncate(self, waveform, length):
        """
        Pads or truncates the waveform to a fixed length.

        Args:
            waveform (torch.Tensor): Audio waveform tensor.
            length (int): Desired length in samples.

        Returns:
            torch.Tensor: Waveform tensor padded or truncated to the specified length.
        """
        if waveform.shape[0] > length:
            # Truncate the waveform
            waveform = waveform[:length]
        elif waveform.shape[0] < length:
            # Pad the waveform with zeros at the end
            padding = length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

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
        # Calculate power of speech and interference
        speech_power = (speech.norm(p=2)) ** 2
        interference_power = (interference.norm(p=2)) ** 2

        # Calculate the scaling factor for interference to achieve desired SNR
        snr_linear = 10 ** (snr_db / 10)
        scaling_factor = speech_power / (interference_power * snr_linear)

        # Scale interference and mix
        interference_scaled = interference * scaling_factor.sqrt()
        mixed = speech + interference_scaled

        return mixed

    def _get_random_file_from_speaker(self, speaker_id):
        """
        Selects a random audio file from the given speaker.

        Args:
            speaker_id (str): The ID of the speaker.

        Returns:
            str: Path to the selected audio file.
        """
        speaker_dir = os.path.join(self.lrs3_root, speaker_id)
        files = [os.path.join(speaker_dir, f) for f in os.listdir(speaker_dir) if f.lower().endswith('.wav')]
        if not files:
            raise ValueError(f"No files found for speaker {speaker_id}")
        return random.choice(files)

    def _create_interfering_waveform(self, mode, lrs3_file=None):
        if mode == 'speaker':
            # Speaker Separation: Add speech from another speaker
            clean_speaker_id = self._extract_speaker_id(lrs3_file)

            # Select a different speaker
            other_speakers = [spk for spk in self.speakers if spk != clean_speaker_id]
            if not other_speakers:
                raise ValueError("No other speakers available for interference.")
            interfering_speaker_id = random.choice(other_speakers)

            # Select a random file from the interfering speaker
            interfering_file = self._get_random_file_from_speaker(interfering_speaker_id)
            interfering_waveform, orig_sample_rate = torchaudio.load(interfering_file)
            interfering_waveform = torchaudio.functional.resample(interfering_waveform, orig_freq=orig_sample_rate,
                                                                  new_freq=self.sample_rate)
            interfering_waveform = interfering_waveform.squeeze(0)  # Assuming mono; adjust if stereo

            # Pad or truncate interfering waveform to fixed length
            interfering_waveform = self.pad_or_truncate(interfering_waveform, self.fixed_length)

            interference_type = 'speaker'


        elif mode == 'noise':

            # Speech Enhancement: Add background noise from DNS

            max_retries = 2

            retries = 0

            interfering_waveform = None

            while retries <= max_retries:

                idx_dns = random.randint(1, self.dns_files_len)
                dns_file = linecache.getline(self.dns_files_list, idx_dns).strip()
                interfering_waveform, orig_sample_rate = torchaudio.load(dns_file)

                # Calculate interference power
                interference_power = (interfering_waveform.norm(p=2)) ** 2

                if interference_power > 0:
                    # Successfully found a valid file
                    break

                # Retry logic if power is zero

                print(f"Retry {retries + 1}/{max_retries}: Interference power is zero for file {dns_file}")
                retries += 1

            if retries > max_retries:
                raise ValueError(f"All retries failed: Unable to find valid DNS file after {max_retries + 1} attempts.")

            # Resample to match the target sample rate

            interfering_waveform = torchaudio.functional.resample(interfering_waveform, orig_freq=orig_sample_rate,

                                                                  new_freq=self.sample_rate)

            interfering_waveform = interfering_waveform.squeeze(0)  # Assuming mono; adjust if stereo

            # Pad or truncate interfering waveform to fixed length

            interfering_waveform = self.pad_or_truncate(interfering_waveform, self.fixed_length)

            interference_type = 'noise'


        else:
            raise ValueError("Invalid mode selected.")

        return interfering_waveform, interference_type

    def _preprocess_audio(self, lrs3_file):
        # Load clean speech from LRS3
        speech_waveform, orig_sample_rate = torchaudio.load(lrs3_file)
        speech_waveform = torchaudio.functional.resample(speech_waveform, orig_freq=orig_sample_rate,
                                                         new_freq=self.sample_rate)
        speech_waveform = speech_waveform.squeeze(0)  # Assuming mono; adjust if stereo

        # Pad or truncate speech waveform to fixed length
        speech_waveform = self.pad_or_truncate(speech_waveform, self.fixed_length)

        # Decide randomly whether to add another speaker or noise
        mode = random.choices(['speaker', 'noise'], weights=[self.mode_prob.get('speaker', 0.5),
                                                             self.mode_prob.get('noise', 0.5)])[0]
        interfering_waveform, interference_type = self._create_interfering_waveform(mode, lrs3_file=lrs3_file)
        # Mix speech and interference at desired SNR
        mixture = self.add_noise_with_snr(speech_waveform, interfering_waveform, self.snr_db)

        # Add channel and batch dimensions before encoding
        mixture = mixture.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, samples]

        # Encode using denoiser's encoder
        with torch.no_grad():
            encoded_audio = mixture
            for layer in self.encoder:
                encoded_audio = layer(encoded_audio)

        # Remove batch dimension after encoding
        encoded_audio = encoded_audio.squeeze(0)  # Shape: [channels, encoded_length]

        return encoded_audio, mixture, speech_waveform, interfering_waveform, interference_type

    def _preprocess_video(self, file_path: str):
        return self.lipreading_preprocessing.generate_encodings(self.video_processor.crop_video_96_96(file_path))

    def __getitem__(self, idx):
        # Read the paired file paths from the text file
        paired_line = linecache.getline(self.paired_files_list, idx + 1).strip()
        if not paired_line:
            raise IndexError(f"No paired files found at index {idx}")

        audio_lrs3_file, video_lrs3_file = paired_line.split('\t')
        # print(f"Processing Audio: {audio_lrs3_file} | Video: {video_lrs3_file}")


        encoded_video = self._preprocess_video(video_lrs3_file)
        encoded_audio, mixture, speech_waveform, interfering_waveform, interference_type = self._preprocess_audio(
            audio_lrs3_file)

        # Transpose encoded_audio from [batch_size, channels, seq_len] to [batch_size, seq_len, channels]
        encoded_audio = encoded_audio.permute(1, 0)  # [batch_size, seq_len, channels]


        sample = {
            'encoded_audio': encoded_audio,
            'encoded_video': encoded_video,  # Shape: [100, 96, 96]
            'clean_speech': speech_waveform,  # Shape: [1, samples]
            'audio_file_path': audio_lrs3_file,
            'video_file_path': video_lrs3_file
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

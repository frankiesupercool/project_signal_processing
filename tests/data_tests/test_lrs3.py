import os
import random
import torch
import torchaudio
from denoiser import pretrained
import config


def add_noise_with_snr(speech, interference, snr_db, epsilon=1e-10):
    """
    Mixes speech with interference (noise or another speaker) at the specified SNR.

    Args:
        speech (torch.Tensor): Clean speech waveform.
        interference (torch.Tensor): Interfering waveform (noise or another speaker's speech).
        snr_db (float): Desired Signal-to-Noise Ratio in decibels.
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        torch.Tensor: Mixed waveform.
    """
    # Calculate power of speech and interference
    speech_power = speech.norm(p=2).pow(2)
    interference_power = interference.norm(p=2).pow(2) + epsilon  # Prevent division by zero

    # Calculate the scaling factor for interference to achieve desired SNR
    snr_linear = 10 ** (snr_db / 10)
    scaling_factor = speech_power / (interference_power * snr_linear)

    # Prevent scaling_factor from becoming too large
    max_scaling = 1e3
    scaling_factor = torch.clamp(scaling_factor, max=max_scaling)

    # Scale interference and mix
    interference_scaled = interference * torch.sqrt(scaling_factor)
    mixed = speech + interference_scaled

    return mixed


def _create_interfering_waveform(mode, lrs3_file, noise_folder):
    if mode == 'speaker':
        # use same file for testing
        interfering_file = lrs3_file
        interfering_waveform, orig_sample_rate = torchaudio.load(interfering_file)
        interfering_waveform = torchaudio.functional.resample(interfering_waveform, orig_freq=orig_sample_rate,
                                                              new_freq=16000)
        interfering_waveform = interfering_waveform.squeeze(0)  # Assuming mono; adjust if stereo

        # Pad or truncate interfering waveform to fixed length
        interfering_waveform = pad_or_truncate(interfering_waveform, 64000)

        interference_type = 'speaker'


    elif mode == 'noise':

        # Speech Enhancement: Add background noise from DNS

        max_retries = 2

        retries = 0

        interfering_waveform = None

        while retries <= max_retries:
            dns_file = random.choice(os.listdir(noise_folder))
            interfering_waveform, orig_sample_rate = torchaudio.load(noise_folder + "/" + dns_file)

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

                                                              new_freq=16000)

        interfering_waveform = interfering_waveform.squeeze(0)  # Assuming mono; adjust if stereo

        # Pad or truncate interfering waveform to fixed length

        interfering_waveform = pad_or_truncate(interfering_waveform, 64000)

        interference_type = 'noise'


    else:
        raise ValueError("Invalid mode selected.")

    return interfering_waveform, interference_type


def pad_or_truncate(waveform, length):
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


def check_mixture(speech_waveform, lrs3_file, noise_folder):
    mode_prob = {'speaker': 0.5, 'noise': 0.5}
    mode = random.choices(['speaker', 'noise'], weights=[mode_prob.get('speaker', 0.5),
                                                         mode_prob.get('noise', 0.5)])[0]

    interfering_waveform, interference_type = _create_interfering_waveform(mode, lrs3_file, noise_folder)
    # Mix speech and interference at desired SNR
    mixture = add_noise_with_snr(speech_waveform, interfering_waveform, 10)

    mixture = mixture.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, samples]
    if mixture.shape != torch.Size([1, 1, 64000]):
        print("Mixture shape not 1,1,sample for", lrs3_file)
    return mixture, interference_type


def check_lrs3_audio(file_path):
    """
    Validate the integrity of an LRS3 audio file.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        bool: True if the audio file is valid, False otherwise.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return False

    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_path)

        # Check if waveform is non-empty
        if waveform.numel() == 0:
            return False

        # Validate sample rate (assumes valid range is 8kHz to 48kHz)
        if not (8000 <= sample_rate <= 48000):
            return False

        # Validate duration (assumes valid duration is >= 0.5 seconds)
        duration = waveform.size(1) / sample_rate
        if duration < 0.5:
            return False

    except Exception:
        return False

    return True


def check_speech_waveform(speech_waveform, orig_sample_rate, file_path):
    if speech_waveform is None or speech_waveform.numel() == 0:
        print(f"Error: Empty or invalid audio file: {file_path}")
        return None

    if orig_sample_rate <= 0:
        print(f"Invalid sample rate: {file_path}")
        return None

    if torch.all(speech_waveform == 0):
        print(f"File contains only silence: {file_path}")
    elif torch.max(torch.abs(speech_waveform)) >= 1.0:
        print(f"Warning: Potential clipping in file: {file_path}")


def check_encode(mixture, encoder, wav_path, interference_type):
    with torch.no_grad():
        encoded_audio = mixture
        for layer in encoder:
            encoded_audio = layer(encoded_audio)

    # Remove batch dimension after encoding
    encoded_audio = encoded_audio.squeeze(0)

    if encoded_audio.numel() == 0:
        print("Error: Encoded audio tensor is empty.")
        print("File:", wav_path)
        print("Interference type:", interference_type)
    elif torch.isnan(encoded_audio).any():
            print("Warning: Encoded audio contains NaN values.")
            print("File:", wav_path)
            print("Interference type:", interference_type)
    elif torch.isinf(encoded_audio).any():
        print("Warning: Encoded audio contains infinite values.")
        print("File:", wav_path)
        print("Interference type:", interference_type)


def test_lrs3_files(folder, noise_folder, encoder):
    """
    Check for corrupted LRS3 files
    """
    count = 0
    for folder_name, subfolders, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                wav_path = os.path.join(folder_name, filename)

                # basic check on LRS3 files
                if not check_lrs3_audio(wav_path):
                    print(f"Error with lrs file {wav_path}")
                else:
                    count += 1

                # check speech_waveform
                speech_waveform, orig_sample_rate = torchaudio.load(wav_path)
                speech_waveform = torchaudio.functional.resample(speech_waveform, orig_freq=orig_sample_rate,
                                                                 new_freq=16000)
                speech_waveform = speech_waveform.squeeze(0)

                check_speech_waveform(speech_waveform, orig_sample_rate, wav_path)

                # pad or truncate check
                speech_waveform = pad_or_truncate(speech_waveform, 64000)

                # check mixture
                mixture, interference_type = check_mixture(speech_waveform, wav_path, noise_folder)

                # check encoder
                check_encode(mixture, encoder, wav_path, interference_type)

    print("Correct file from basic check:", count)


if __name__ == "__main__":
    """
    Pretest LRS3 data set for corrupt files, walk through preprocessing steps to identify broken files
    Uses old denoiser setup!!!
    """
    noise_folder = config.DNS_DATA_PATH
    root_folders = {config.PRETRAIN_DATA_PATH, config.TEST_DATA_PATH, config.TRAINVAL_DATA_PATH}
    
    # setup encoder (old version)
    model = pretrained.dns64()
    encoder = model.encoder

    for root_folder in root_folders:
        test_lrs3_files(root_folder, noise_folder, encoder)

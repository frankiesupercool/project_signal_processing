import os
import torchaudio
import config


def test_wav_files(wav_folder):
    """
    Precheck all WAV files in a folder to ensure they are valid, have non-zero power, and are not in unsigned 8 bit
    format (PCM_U)
    Args:
        wav_folder: Path to the folder containing WAV files.

    Returns: List of valid WAV files.

    """
    valid_files = []
    for file in os.listdir(wav_folder):
        if file.lower().endswith('.wav'):  # Check for WAV files
            wav_path = os.path.join(wav_folder, file)
            try:
                metadata = torchaudio.info(wav_path)
                encoding = metadata.encoding

                # check bits_per_sample=8
                if "PCM_U" in encoding:
                    print(f"File {wav_path} is in 'Unsigned 8 bit' format and will be skipped.")
                    continue

                waveform, _ = torchaudio.load(wav_path)

                interference_power = (waveform.norm(p=2)) ** 2

                if interference_power > 0:
                    valid_files.append(wav_path)
                else:
                    print(f"File {wav_path} has zero interference power and will be skipped.")
            except Exception as e:
                print(f"Error processing file {wav_path}: {e}")
    return valid_files


if __name__ == "__main__":
    noise_folder = config.DNS_DATA_PATH
    test_wav_files(noise_folder)

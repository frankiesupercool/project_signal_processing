import matplotlib.pyplot as plt
import os

import config
from dataset_lightning.dataset import PreprocessingDataset
from torch.utils.data import DataLoader

# Ensure that the PreprocessingDataset class is defined above this script
# If it's in a separate module, you can import it using:
# from your_module import PreprocessingDataset


def test_preprocessing_dataset():
    # Define paths to your x_datasets
    lrs3_root = config.PRETRAIN_DATA_PATH # Replace with your actual LRS3 root directory
    dns_root = config.DNS_DATA_PATH    # Replace with your actual DNS root directory

    # Check if directories exist
    if not os.path.isdir(lrs3_root):
        raise FileNotFoundError(f"LRS3 root directory not found: {lrs3_root}")
    if not os.path.isdir(dns_root):
        raise FileNotFoundError(f"DNS root directory not found: {dns_root}")

    # Instantiate the dataset
    dataset = PreprocessingDataset(
        lrs3_root=lrs3_root,
        dns_root=dns_root,
        densetcn_options=config.densetcn_options,
        allow_size_mismatch=config.allow_size_mismatch,
        model_path= config.MODEL_PATH,
        use_boundary=config.use_boundary,
        relu_type=config.relu_type,
        num_classes=config.num_classes,
        backbone_type=config.backbone_type,
        snr_db=config.snr_db,  # Example SNR value
        sample_rate=config.sample_rate,
        fixed_length=config.fixed_length
    )

    # Create a DataLoader for batching (optional)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    print(f"Dataset Length: {len(dataset)}")

    # Fetch a single batch
    for batch_idx, sample_batch in enumerate(dataloader):
        print(f"\n--- Batch {batch_idx + 1} ---")

        # Extract batch components
        encoded_audio = sample_batch['encoded_audio']        # Shape: [batch_size, channels, encoded_length]
        encoded_video = sample_batch['encoded_video']        # Shape: [batch_size, channels, encoded_length]
        clean_speech = sample_batch['clean_speech']          # Shape: [batch_size, 1, samples]         # Shape: [batch_size, 1, samples]
        audio_file_paths = sample_batch['audio_file_path']   # List of file paths
        video_file_paths = sample_batch['video_file_path']      # Shape: [batch_size, frames, 96, 96]

        # Print shapes and types
        print(f"Encoded Audio Shape: {encoded_audio.shape}")
        print(f"Clean Speech Shape: {clean_speech.shape}")

        # Verify that audio and video files correspond
        for i in range(len(audio_file_paths)):
            audio_path = audio_file_paths[i]
            # Assuming the video file has the same base name with .mp4 extension
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            video_path = os.path.join(os.path.dirname(audio_path), f"{base_name}.mp4")
            if not os.path.exists(video_path):
                print(f"Error: Corresponding video file does not exist for {audio_path}")
            else:
                print(f"Audio: {audio_path} <--> Video: {video_path}")

        # (Optional) Visualize the first sample in the batch
        #visualize_sample(
            #sample_batch,
            #index=0,
            #audio_sample_rate=dataset.sample_rate
        #)

        # For testing purposes, we'll just process one batch
        break

def visualize_sample(sample_batch, index=0, audio_sample_rate=16000):
    """
    Visualizes the audio waveform and video frames of a single sample.

    Args:
        sample_batch (dict): A batch from the DataLoader.
        index (int): Index of the sample in the batch to visualize.
        audio_sample_rate (int): Sample rate of the audio for plotting.
    """
    import numpy as np

    # Extract sample data         # Shape: [samples]
    clean_speech = sample_batch['clean_speech'][index].squeeze().numpy() # Shape: [samples]
    interference = sample_batch['interference'][index].squeeze().numpy() # Shape: [samples]
    video_sequence = sample_batch['video_sequence'][index].numpy()       # Shape: [frames, 96, 96]
    interference_type = sample_batch['interference_type'][index]
    audio_file_path = sample_batch['audio_file_path'][index]
    print(f"\nVisualizing Sample:")
    print(f"Audio File: {audio_file_path}")
    print(f"Interference Type: {interference_type}")

    # Plot audio waveforms
    plt.figure(figsize=(15, 6))


    plt.subplot(3, 1, 2)
    plt.title("Clean Speech Waveform")
    plt.plot(clean_speech)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 3)
    plt.title(f"Interference Waveform ({interference_type})")
    plt.plot(interference)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

    # Display a few video frames
    num_frames_to_show = 5
    plt.figure(figsize=(15, 3))
    for i in range(num_frames_to_show):
        plt.subplot(1, num_frames_to_show, i + 1)
        plt.imshow(video_sequence[i], cmap='gray')
        plt.title(f"Frame {i + 1}")
        plt.axis('off')
    plt.suptitle("Video Frames")
    plt.show()

if __name__ == "__main__":
    test_preprocessing_dataset()
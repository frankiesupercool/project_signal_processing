import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import PreprocessingDataset
from lightning_datamodule import DataModule
import torch

def test_data_module():
    # Initialize the DataModule
    lrs3_root = '../data/pretrain'  # Replace with actual path
    dns_root = '../noise_data_set/noise'    # Replace with actual path

    data_module = DataModule(
        lrs3_root=lrs3_root,
        dns_root=dns_root,
        snr_db=0,
        transform=None,
        sample_rate=16000,
        mode_prob={'speaker': 0.5, 'noise': 0.5},
        batch_size=4,
        num_workers=0  # Start with 0 for testing
    )

    # Prepare the DataModule
    data_module.setup()

    data_module.num_workers = 4  # Increase as per your CPU cores

    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    # Print the shapes of 'encoded_audio' and 'clean_speech'
    print("Encoded Audio Shape:", batch['encoded_audio'].shape)
    print("Clean Speech Shape:", batch['clean_speech'].shape)

    # Inspect the batch
    print("Batch keys:", batch.keys())
    print("Encoded audio shape:", batch['encoded_audio'].shape)
    print("Mixture shape:", batch['mixture'].shape)
    print("Clean speech shape:", batch['clean_speech'].shape)
    print("Interference shape:", batch['interference'].shape)
    print("Interference type:", batch['interference_type'])
    print("File paths:", batch['file_path'])

    import matplotlib.pyplot as plt

    def plot_waveform(waveform, sample_rate, title="Waveform"):
        waveform = waveform.numpy()
        time_axis = torch.arange(0, waveform.shape[-1]) / sample_rate
        plt.figure()
        plt.plot(time_axis, waveform)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

    # Plot the first sample in the batch
    sample_idx = 0
    plot_waveform(batch['mixture'][sample_idx].squeeze(), data_module.sample_rate, title="Mixture")
    plot_waveform(batch['clean_speech'][sample_idx].squeeze(), data_module.sample_rate, title="Clean Speech")
    plot_waveform(batch['interference'][sample_idx].squeeze(), data_module.sample_rate, title="Interference")


if __name__ == "__main__":
    test_data_module()

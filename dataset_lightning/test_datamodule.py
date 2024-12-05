import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import PreprocessingDataset
from lightning_datamodule import DataModule
import torch

def test_data_module():
    # Initialize the DataModule general
    lrs3_root = '../data/pretrain'  # Replace with actual path

    # Initialisation values for audio
    dns_root = '../noise_data_set/noise'    # Replace with actual path

    # Initialisation values for video
    densetcn_options = {'block_config': [3,
                                         3,
                                         3,
                                         3],
                        'growth_rate_set': [384,
                                            384,
                                            384,
                                            384],
                        'reduced_size': 512,
                        'kernel_size_set': [3,
                                            5,
                                            7],
                        'dilation_size_set': [1,
                                              2,
                                              5],
                        'squeeze_excitation': True,
                        'dropout': 0.2,
                        }

    allow_size_mismatch = False
    model_path = '../video_encoding/lrw_resnet18_dctcn_video_boundary.pth'
    use_boundary = True
    relu_type = "swish"
    num_classes = 500
    backbone_type = "resnet"


    data_module = DataModule(
        lrs3_root=lrs3_root,
        dns_root=dns_root,
        snr_db=0,
        transform=None,
        sample_rate=16000,
        mode_prob={'speaker': 0.5, 'noise': 0.5},
        batch_size=4,
        num_workers=0, # Start with 0 for testing
        densetcn_options=densetcn_options,
        allow_size_mismatch=allow_size_mismatch,
        model_path=model_path,
        use_boundary=use_boundary,
        relu_type=relu_type,
        num_classes=num_classes,
        backbone_type=backbone_type,
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

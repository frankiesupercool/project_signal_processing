import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset_lightning.lightning_datamodule import DataModule
import torch
import os

def test_data_module():
    # Define dataset paths
    pretrain_root = '../data/pretrain'   # Path for pretraining data
    trainval_root = '../data/trainval'   # Path for training-validation data
    test_root = '../data/test'           # Path for testing data
    dns_root = '../noise_data_set/noise' # Path for DNS noise data

    # Verify that directories exist
    for path in [pretrain_root, trainval_root, test_root, dns_root]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Required directory not found: {path}")

    # Video encoding options
    densetcn_options = {
        'block_config': [3, 3, 3, 3],
        'growth_rate_set': [384, 384, 384, 384],
        'reduced_size': 512,
        'kernel_size_set': [3, 5, 7],
        'dilation_size_set': [1, 2, 5],
        'squeeze_excitation': True,
        'dropout': 0.2,
    }

    # Model and processing configurations
    allow_size_mismatch = True
    model_path = '../video_encoding/lrw_resnet18_dctcn_video_boundary.pth'
    use_boundary = True
    relu_type = "swish"
    num_classes = 500
    backbone_type = "resnet"

    # Initialize the DataModule with correct parameters
    data_module = DataModule(
        pretrain_root=pretrain_root,
        trainval_root=trainval_root,
        test_root=test_root,
        dns_root=dns_root,
        densetcn_options=densetcn_options,
        allow_size_mismatch=allow_size_mismatch,
        model_path=model_path,
        use_boundary=use_boundary,
        relu_type=relu_type,
        num_classes=num_classes,
        backbone_type=backbone_type,
        snr_db=0,
        transform=None,
        sample_rate=16000,
        mode_prob={'speaker': 0.5, 'noise': 0.5},
        batch_size=4,
        num_workers=4,  # Adjust based on your CPU cores
        fixed_length=64000,
        fixed_frames=100,
        seed=42,
    )

    # Prepare the DataModule (this will set up datasets)
    data_module.setup()

    # Fetch DataLoaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Example: Fetch a batch from the training DataLoader
    test_batch = next(iter(train_loader))

    # Print batch details
    print("Batch keys:", test_batch.keys())
    print("Encoded Audio Shape:", test_batch['encoded_audio'].shape)
    print("Encoded Video Shape:", test_batch['encoded_video'].shape)
    print("Clean Speech Shape:", test_batch['clean_speech'].shape)
    print("Audio File Paths:", test_batch['audio_file_path'])
    print("Video File Paths:", test_batch['video_file_path'])

    # Example: Fetch a batch from the training DataLoader
    val_batch = next(iter(val_loader))

    # Print batch details
    print("Batch keys:", val_batch.keys())
    print("Encoded Audio Shape:", val_batch['encoded_audio'].shape)
    print("Encoded Video Shape:", val_batch['encoded_video'].shape)
    print("Clean Speech Shape:", val_batch['clean_speech'].shape)
    print("Audio File Paths:", val_batch['audio_file_path'])
    print("Video File Paths:", val_batch['video_file_path'])



if __name__ == "__main__":
    test_data_module()

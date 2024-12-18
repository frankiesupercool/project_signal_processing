import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset_lightning.lightning_datamodule import DataModule
import torch
import os
import config

def test_data_module():
    # Define dataset paths
    pretrain_root = config.PRETRAIN_DATA_PATH   # Path for pretraining data
    trainval_root = config.TRAINVAL_DATA_PATH  # Path for training-validation data
    test_root = config.TEST_DATA_PATH          # Path for testing data
    dns_root = config.DNS_DATA_PATH # Path for DNS noise data

    # Verify that directories exist
    for path in [pretrain_root, trainval_root, test_root, dns_root]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Required directory not found: {path}")


    # Initialize the DataModule with correct parameters
    data_module = DataModule(
        pretrain_root=pretrain_root,
        trainval_root=trainval_root,
        test_root=test_root,
        dns_root=dns_root,
        densetcn_options=config.densetcn_options,
        allow_size_mismatch=config.allow_size_mismatch,
        model_path=config.MODEL_PATH,
        use_boundary=config.use_boundary,
        relu_type=config.relu_type,
        num_classes=config.num_classes,
        backbone_type=config.backbone_type,
        snr_db=config.snr_db,
        transform=None,
        sample_rate=config.sample_rate,
        mode_prob=config.mode_prob,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        fixed_length=config.fixed_length,
        fixed_frames=config.fixed_frames,
        seed=config.SEED,
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

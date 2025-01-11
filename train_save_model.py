import pytorch_lightning as pl
import torch.nn as nn
import torch
import os
from pytorch_lightning.callbacks import ModelCheckpoint

import config
from dataset_lightning.lightning_datamodule import DataModule
from transformer.AV_transformer import AudioVideoTransformer
from transformer.transformer_model import TransformerModel

def train():
    # Define dataset paths
    pretrain_root = config.PRETRAIN_DATA_PATH     # Path for pretraining data
    trainval_root = config.TRAINVAL_DATA_PATH      # Path for training-validation data
    test_root = config.TEST_DATA_PATH           # Path for testing data
    dns_root = config.DNS_DATA_PATH   # Path for DNS noise data

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

    # Fetch a single batch to determine encoded_length and channels
    sample_batch = next(iter(train_loader))
    encoded_audio_shape = sample_batch['encoded_audio'].shape  # [batch_size, channels, encoded_length]
    encoded_length = encoded_audio_shape[2]  # Extract encoded_length
    channels = encoded_audio_shape[1]         # Extract number of channels
    # print(f"Encoded Audio Shape: {encoded_audio_shape}")
    # print(f"Determined encoded_length: {encoded_length}")
    # print(f"Number of Channels: {channels}")

    # Initialize the model with the determined encoded_length and channels
    transformer_model_instance = TransformerModel(
        audio_dim=1024,  # From your encoded_audio
        video_dim=500,  # From your encoded_video
        embed_dim=768,  # As per your specification
        nhead=8,        # As per your specification
        num_layers=3,   # As per your specification
        dim_feedforward=532,  # As per your specification
        max_seq_length=1024,  # Adjust based on your sequence lengths
        denoiser_decoder=None  # Provide the denoiser's decoder if available
    )
    model = AudioVideoTransformer(model=transformer_model_instance)

    # Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',               # Metric to monitor
        dirpath='checkpoints',            # Directory to save checkpoints
        filename='best-checkpoint',       # Filename template
        save_top_k=1,                      # Save only the best model
        mode='min',                        # Mode: 'min' for val_loss
        verbose=True                       # Verbose logging
    )

    # Initialize the Trainer with updated arguments
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            max_epochs=5,
            accelerator='gpu',
            devices=1,
            callbacks=[checkpoint_callback],    # Add the checkpoint callback here
            log_every_n_steps=1,                # To see logs for each training step
        )
    else:
        trainer = pl.Trainer(
            max_epochs=5,
            accelerator='cpu',
            callbacks=[checkpoint_callback],    # Add the checkpoint callback here
            log_every_n_steps=1,                # To see logs for each training step
        )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Optionally, print the path to the best checkpoint
    print("Training complete!")
    print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    train()

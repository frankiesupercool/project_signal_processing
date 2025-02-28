import pytorch_lightning as pl
import torch.nn as nn
import torch

import config
from dataset_lightning.lightning_datamodule import DataModule  # Adjusted import path
import os

class DummyModel(pl.LightningModule):
    def __init__(self, encoded_length, channels):
        super().__init__()
        self.encoded_length = encoded_length
        self.channels = channels
        self.input_size = self.channels * self.encoded_length
        print(f"Initializing DummyModel with channels={self.channels}, encoded_length={self.encoded_length}, input_size={self.input_size}")
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, 1)  # Predict a single value
        )

    def training_step(self, batch, batch_idx):
        x = batch['encoded_audio']  # [batch_size, channels, encoded_length]
        y = batch['clean_speech']   # [batch_size, 1, 64000]

        # Forward pass (Flattening is handled by the model)
        pred = self.model(x)

        # Compute the mean of clean_speech
        y_mean = y.view(y.size(0), -1).mean(dim=1, keepdim=True)

        # Compute loss
        loss = torch.mean((pred - y_mean) ** 2)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['encoded_audio']  # [batch_size, channels, encoded_length]
        y = batch['clean_speech']   # [batch_size, 1, 64000]

        # Forward pass (Flattening is handled by the model)
        pred = self.model(x)

        # Compute the mean of clean_speech
        y_mean = y.view(y.size(0), -1).mean(dim=1, keepdim=True)

        # Compute loss
        loss = torch.mean((pred - y_mean) ** 2)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def test_trainer():
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
        snr_db=config.snr_db,
        sample_rate=config.sample_rate,
        mode_prob=config.mode_prob,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        fixed_length=config.fixed_length,
        fixed_frames=config.fixed_frames,
    )

    # Prepare the DataModule (this will set up datasets)
    data_module.setup()

    # Fetch DataLoaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Fetch a single batch to determine encoded_length and channels
    sample_batch = next(iter(train_loader))
    encoded_audio_shape = sample_batch['encoded_audio'].shape  # [batch_size, channels, encoded_length]
    encoded_length = encoded_audio_shape[2]  # Extract encoded_length
    channels = encoded_audio_shape[1]         # Extract number of channels
    print(f"Encoded Audio Shape: {encoded_audio_shape}")
    print(f"Determined encoded_length: {encoded_length}")
    print(f"Number of Channels: {channels}")

    # Initialize the model with the determined encoded_length and channels
    model = DummyModel(encoded_length=encoded_length, channels=channels)

    # Initialize the Trainer with updated arguments
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=2,          # Limit for quick testing
            accelerator='gpu',
            devices=1,
            log_every_n_steps=1,            # To see logs for each training step
        )
    else:
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=2,          # Limit for quick testing
            accelerator='cpu',
            log_every_n_steps=1,            # To see logs for each training step
        )

    # Train the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    test_trainer()



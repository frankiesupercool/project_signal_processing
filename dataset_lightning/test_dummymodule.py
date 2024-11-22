import pytorch_lightning as pl
import torch.nn as nn
import torch
from lightning_datamodule import DataModule

import pytorch_lightning as pl
import torch.nn as nn
import torch

class DummyModel(pl.LightningModule):
    def __init__(self, encoded_length):
        super().__init__()
        # Reduce input size using a simple layer
        self.input_size = 1024 * encoded_length
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, 1)  # Predict a single value
        )

    def training_step(self, batch, batch_idx):
        x = batch['encoded_audio']  # [batch_size, 1024, encoded_length]
        y = batch['clean_speech']   # [batch_size, 1, 64000]

        # Flatten x
        x = x.view(x.size(0), -1)

        # Forward pass
        pred = self.model(x)

        # Compute the mean of clean_speech
        y_mean = y.view(y.size(0), -1).mean(dim=1, keepdim=True)

        # Compute loss
        loss = torch.mean((pred - y_mean) ** 2)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)




def test_trainer():
    lrs3_root = '../data/pretrain'  # Replace with actual path
    dns_root = '../noise_data_set/noise'
    # Initialize the DataModule
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

    # Initialize the model
    model = DummyModel(61)

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,  # Limit for quick testing
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    test_trainer()

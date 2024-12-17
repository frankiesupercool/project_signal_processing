import pytorch_lightning as pl
import torch.nn as nn
import torch
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
    pretrain_root = '../../../../../data/LRS3/pretrain'      # Path for pretraining data
    trainval_root = '../../../../../data/LRS3/trainval'      # Path for training-validation data
    test_root = '../../../../../data/LRS3/test'              # Path for testing data
    dns_root = '../noise_data_set/noise'    # Path for DNS noise data

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



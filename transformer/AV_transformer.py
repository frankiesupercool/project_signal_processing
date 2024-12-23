import pytorch_lightning as pl
from torch import nn, optim
import torch

class AudioVideoTransformer(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super(AudioVideoTransformer, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()  # Using Mean Squared Error loss

    def forward(self, encoded_audio, encoded_video):
        """
        Forward pass through the underlying model.
        Expects:
          - encoded_audio: shape [batch_size, seq_len, audio_dim]
          - encoded_video: shape [batch_size, seq_len, video_dim]
        Returns:
          - predicted_clean: shape [batch_size, 64000] (example)
        """
        return self.model(encoded_audio, encoded_video)

    def training_step(self, batch, batch_idx):
        """
        Training step:
         1) forward pass
         2) calculate loss
         3) log 'train_loss'
        """
        encoded_audio = batch['encoded_audio']
        encoded_video = batch['encoded_video']
        clean_speech = batch['clean_speech']

        predicted_clean = self(encoded_audio, encoded_video)
        loss = self.criterion(predicted_clean, clean_speech)

        # Log the training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step:
         1) forward pass
         2) calculate loss
         3) log 'val_loss'
        """
        encoded_audio = batch['encoded_audio']
        encoded_video = batch['encoded_video']
        clean_speech = batch['clean_speech']

        predicted_clean = self(encoded_audio, encoded_video)
        loss = self.criterion(predicted_clean, clean_speech)

        # Log the validation loss
        # IMPORTANT: 'val_loss' must match the monitor name in callbacks
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step:
         1) forward pass
         2) calculate loss
         3) log 'test_loss'
        """
        encoded_audio = batch['encoded_audio']
        encoded_video = batch['encoded_video']
        clean_speech = batch['clean_speech']

        predicted_clean = self(encoded_audio, encoded_video)
        loss = self.criterion(predicted_clean, clean_speech)

        # Log the test loss
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer. Feel free to change optimizer & learning rate if needed.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


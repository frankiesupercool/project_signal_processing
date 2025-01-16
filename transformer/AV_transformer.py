import pytorch_lightning as pl
from torch import nn, optim
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class AudioVideoTransformer(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-5):
        super(AudioVideoTransformer, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()  # Using Mean Squared Error loss

        # Initialize a counter for skipped batches
        self.skipped_batches = 0

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
         1) Check for NaNs or Infs in inputs and targets.
         2) If found, log and skip the batch.
         3) Otherwise, perform forward pass, compute loss, and log 'train_loss'.
        """
        encoded_audio = batch['encoded_audio']
        encoded_video = batch['encoded_video']
        clean_speech = batch['clean_speech']

        # Check for NaNs or Infs in inputs and targets
        if not torch.isfinite(encoded_audio).all():
            logger.warning(f"NaNs or Infs found in encoded_audio at batch {batch_idx}. Skipping batch.")
            self.skipped_batches += 1
            return torch.tensor(0.0, device=self.device, requires_grad=True)  # Dummy loss
        if not torch.isfinite(encoded_video).all():
            logger.warning(f"NaNs or Infs found in encoded_video at batch {batch_idx}. Skipping batch.")
            self.skipped_batches += 1
            return torch.tensor(0.0, device=self.device, requires_grad=True)  # Dummy loss
        if not torch.isfinite(clean_speech).all():
            logger.warning(f"NaNs or Infs found in clean_speech at batch {batch_idx}. Skipping batch.")
            self.skipped_batches += 1
            return torch.tensor(0.0, device=self.device, requires_grad=True)  # Dummy loss

        # Proceed with forward pass and loss computation
        predicted_clean = self(encoded_audio, encoded_video)
        loss = self.criterion(predicted_clean, clean_speech)

        # Determine batch size
        batch_size = encoded_audio.shape[0]

        # Log the training loss with explicit batch_size
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size  # Explicitly specify batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step:
         1) Check for NaNs or Infs in inputs and targets.
         2) If found, log and skip the batch.
         3) Otherwise, perform forward pass, compute loss, and log 'val_loss'.
        """
        encoded_audio = batch['encoded_audio']
        encoded_video = batch['encoded_video']
        clean_speech = batch['clean_speech']

        # Check for NaNs or Infs in inputs and targets
        if not torch.isfinite(encoded_audio).all():
            logger.warning(f"NaNs or Infs found in encoded_audio at validation batch {batch_idx}. Skipping batch.")
            self.skipped_batches += 1
            return
        if not torch.isfinite(encoded_video).all():
            logger.warning(f"NaNs or Infs found in encoded_video at validation batch {batch_idx}. Skipping batch.")
            self.skipped_batches += 1
            return
        if not torch.isfinite(clean_speech).all():
            logger.warning(f"NaNs or Infs found in clean_speech at validation batch {batch_idx}. Skipping batch.")
            self.skipped_batches += 1
            return

        # Proceed with forward pass and loss computation
        predicted_clean = self(encoded_audio, encoded_video)
        loss = self.criterion(predicted_clean, clean_speech)

        # Determine batch size
        batch_size = encoded_audio.shape[0]

        # Log the validation loss with explicit batch_size
        self.log(
            'val_loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size  # Explicitly specify batch_size
        )

        # Debugging: Print val_loss
        logger.info(f"Validation Step {batch_idx}: val_loss = {loss.item()}")
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step:
         1) Check for NaNs or Infs in inputs and targets.
         2) If found, log and skip the batch.
         3) Otherwise, perform forward pass, compute loss, and log 'test_loss'.
        """
        encoded_audio = batch['encoded_audio']
        encoded_video = batch['encoded_video']
        clean_speech = batch['clean_speech']

        # Check for NaNs or Infs in inputs and targets
        if not torch.isfinite(encoded_audio).all():
            logger.warning(f"NaNs or Infs found in encoded_audio at test batch {batch_idx}. Skipping batch.")
            self.skipped_batches += 1
            return
        if not torch.isfinite(encoded_video).all():
            logger.warning(f"NaNs or Infs found in encoded_video at test batch {batch_idx}. Skipping batch.")
            self.skipped_batches += 1
            return
        if not torch.isfinite(clean_speech).all():
            logger.warning(f"NaNs or Infs found in clean_speech at test batch {batch_idx}. Skipping batch.")
            self.skipped_batches += 1
            return

        # Proceed with forward pass and loss computation
        predicted_clean = self(encoded_audio, encoded_video)
        loss = self.criterion(predicted_clean, clean_speech)

        # Determine batch size
        batch_size = encoded_audio.shape[0]

        # Log the test loss with explicit batch_size
        self.log(
            'test_loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size  # Explicitly specify batch_size
        )
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer. Feel free to change optimizer & learning rate if needed.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch. Logs the number of skipped batches.
        """
        if self.skipped_batches > 0:
            logger.warning(f"Total skipped training batches this epoch: {self.skipped_batches}")
            # Reset the counter for the next epoch
            self.skipped_batches = 0

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch. Logs the number of skipped batches.
        """
        if self.skipped_batches > 0:
            logger.warning(f"Total skipped validation batches this epoch: {self.skipped_batches}")
            # Reset the counter for the next epoch
            self.skipped_batches = 0

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch. Logs the number of skipped batches.
        """
        if self.skipped_batches > 0:
            logger.warning(f"Total skipped test batches this epoch: {self.skipped_batches}")
            # Reset the counter for the next epoch
            self.skipped_batches = 0


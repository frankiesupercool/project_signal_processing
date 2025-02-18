# train_av.py
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchmetrics import MeanMetric, MinMetric, SignalDistortionRatio
import logging
import os
import glob

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_latest_checkpoint(checkpoint_dir):
    """Use latest saved checkpoint to resume training."""
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")), key=os.path.getmtime, reverse=True)
    return checkpoints[0] if checkpoints else None


class AVTransformerLightningModule(pl.LightningModule):
    def __init__(self, net: nn.Module, learning_rate: float = 1e-5,
                 optimizer_class=optim.Adam,
                 scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau):
        """
        LightningModule that wraps the VoiceFormer model.

        Args:
            net (nn.Module): The audio–video fusion model.
            learning_rate (float): Learning rate.
            optimizer_class: Optimizer class (default: Adam).
            scheduler_class: LR scheduler class (default: ReduceLROnPlateau).
        """
        super().__init__()
        # Save hyperparameters (excluding the network itself)
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.learning_rate = learning_rate
        self.criterion = nn.L1Loss()  # L1Loss as in the original VoiceFormer

        # Metrics (using torchmetrics)
        self.train_loss_metric = MeanMetric()
        self.val_loss_metric = MeanMetric()
        self.test_loss_metric = MeanMetric()
        self.val_loss_best = MinMetric()
        # Optional: signal distortion ratio metric (or any other audio metric)
        self.sdr = SignalDistortionRatio()

    def forward(self, encoded_audio, encoded_video):
        """
        Forward pass through the underlying model.
        Expects:
          - encoded_audio: Tensor of shape [batch, 1, time]
          - encoded_video: Tensor of shape [batch, video_seq, video_feature_dim]
        Returns:
          - predicted_clean: Tensor of shape [batch, waveform_length]
        """
        return self.net(encoded_audio, encoded_video)

    def step(self, batch):
        """
        Common step for train/validation/test.
        Expects batch to be a dictionary with keys:
            'encoded_audio', 'encoded_video', 'clean_speech'
        """
        encoded_audio = batch['encoded_audio']
        encoded_video = batch['encoded_video']
        clean_speech = batch['clean_speech']
        # Forward pass through the model
        predictions = self(encoded_audio, encoded_video)
        loss = self.criterion(predictions, clean_speech)
        return loss, predictions, clean_speech

    def training_step(self, batch, batch_idx):
        # Check for NaNs/Infs (optional)
        encoded_audio = batch['encoded_audio']
        encoded_video = batch['encoded_video']
        clean_speech = batch['clean_speech']
        if not torch.isfinite(encoded_audio).all():
            logger.warning(f"NaNs/Infs in encoded_audio at training batch {batch_idx}.")
            return torch.tensor(0.0, requires_grad=True)
        if not torch.isfinite(encoded_video).all():
            logger.warning(f"NaNs/Infs in encoded_video at training batch {batch_idx}.")
            return torch.tensor(0.0, requires_grad=True)
        if not torch.isfinite(clean_speech).all():
            logger.warning(f"NaNs/Infs in clean_speech at training batch {batch_idx}.")
            return torch.tensor(0.0, requires_grad=True)

        loss, _, _ = self.step(batch)
        self.train_loss_metric.update(loss)
        batch_size = encoded_audio.shape[0]
        self.log("train/loss", self.train_loss_metric, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        self.val_loss_metric.update(loss)
        batch_size = batch['encoded_audio'].shape[0]
        self.log("val/loss", self.val_loss_metric, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=batch_size)
        logger.info(f"Validation Step {batch_idx}: val_loss = {loss.item()}")
        return loss

    def test_step(self, batch, batch_idx):
        loss, predictions, targets = self.step(batch)
        self.test_loss_metric.update(loss)
        batch_size = batch['encoded_audio'].shape[0]
        self.log("test/loss", self.test_loss_metric, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_class(self.parameters(),
                                                   lr=self.learning_rate,
                                                   weight_decay=0.0001)
        scheduler = self.hparams.scheduler_class(optimizer,
                                                 mode='min',
                                                 factor=0.9,
                                                 patience=1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }

    def on_train_epoch_end(self):
        logger.info(f"Training epoch ended. Train loss: {self.train_loss_metric.compute().item()}")
        self.train_loss_metric.reset()

    def on_validation_epoch_end(self):
        val_loss = self.val_loss_metric.compute()
        self.val_loss_best.update(val_loss)
        logger.info(f"Validation epoch ended. Val loss: {val_loss.item()} | Best val loss: {self.val_loss_best.compute().item()}")
        self.val_loss_metric.reset()

    def on_test_epoch_end(self):
        logger.info(f"Test epoch ended. Test loss: {self.test_loss_metric.compute().item()}")
        self.test_loss_metric.reset()
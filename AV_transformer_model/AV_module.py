import logging
import numpy as np
import torch
from torch import nn, optim
from torchmetrics import MeanMetric, MinMetric, SignalDistortionRatio
import pytorch_lightning as pl
from pystoi import stoi
import config

# Setup logger
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class AVTransformerLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float = 5e-5):
        """
        LightningModule that wraps the AV transformer model
        :param model: The AV transformer model
        :param learning_rate: Learning rate (default is 5e-5)
        """
        super().__init__()
        # Save hyperparameters excluding the model itself
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.L1Loss()  # L1Loss as in paper

        # Metrics init
        self.train_loss_metric = MeanMetric()
        self.val_loss_metric = MeanMetric()
        self.test_loss_metric = MeanMetric()
        self.val_loss_best = MinMetric()
        self.test_sdr_metric = SignalDistortionRatio()
        self.test_stoi_values = []
        self.sdr = SignalDistortionRatio()

    def forward(self, encoded_audio, encoded_video):
        """
        Forward pass through the model
        :param encoded_audio: Tensor of shape [batch, 1, time]
        :param encoded_video: Tensor of shape [batch, video_seq, video_feature_dim]
        :return: Predicted clean audio as tensor of shape [batch, waveform_length]
        """
        return self.model(encoded_audio, encoded_video)

    def step(self, batch):
        """
        Common step for train/validation/test phases
        :param batch: Dictionary with keys: 'encoded_audio', 'encoded_video', 'clean_speech'
        :return:
            loss: step loss
            prediction: predicted clean audio
            clean_speech: original clean speech audio
        """

        encoded_audio = batch['encoded_audio']
        encoded_video = batch['encoded_video']
        clean_speech = batch['clean_speech']
        # Calls forward pass through the model
        prediction = self(encoded_audio, encoded_video)
        loss = self.criterion(prediction, clean_speech)
        return loss, prediction, clean_speech

    def training_step(self, batch, batch_idx):
        """
        Training step, checks first for NaNs/Infs in batch and sets them to 0.0
        Calls step, logs loss
        :param batch: Dictionary with 'encoded_audio', 'encoded_video', 'clean_speech'
        :param batch_idx: Index of current batch
        :return: Traning step loss of current batch
        """

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
        self.log("train_loss", self.train_loss_metric, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step calls step, logs loss
        :param batch: Dictionary with 'encoded_audio', 'encoded_video', 'clean_speech'
        :param batch_idx: Index of current batch
        :return: Validation step loss of current batch
        """
        loss, _, _ = self.step(batch)
        self.val_loss_metric.update(loss)
        batch_size = batch['encoded_audio'].shape[0]
        self.log("val_loss", self.val_loss_metric, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=batch_size)
        logger.info(f"Validation Step {batch_idx}: val_loss = {loss.item()}")
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step calls step, logs loss and computes SDR and STOI metrics
        :param batch: Dictionary with 'encoded_audio', 'encoded_video', 'clean_speech'
        :param batch_idx: Index of current batch
        :return: Test step loss of current batch and
        """
        loss, predictions, targets = self.step(batch)
        self.test_loss_metric.update(loss)
        batch_size = batch['encoded_audio'].shape[0]
        self.log("test_loss", self.test_loss_metric, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=batch_size)

        # [batch size, 1, time] - squeeze the channel dimension
        pred_audio = predictions.squeeze(1)
        target_audio = targets.squeeze(1)

        # SDR
        sdr_val = self.test_sdr_metric(pred_audio, target_audio)
        self.log("test_SDR", sdr_val, on_epoch=True, prog_bar=True)

        # STOI for each sample
        for pred, target in zip(pred_audio, target_audio):
            pred_np = pred.cpu().float().numpy().astype(np.float64).flatten()
            target_np = target.cpu().float().numpy().astype(np.float64).flatten()
            stoi_val = stoi(target_np, pred_np, config.sample_rate, extended=False)
            self.test_stoi_values.append(stoi_val)

        return {"loss": loss, "SDR": sdr_val}

    def configure_optimizers(self):
        """
        Sets up an Adam optimizer with weight decay and a ReduceLROnPlateau scheduler to adjust the learning rate
        based on validation loss.
        Reacts quick to being stuck on optima by decreasing learning rate by 10% (factor 0.9)
        after no improvement of 1 epoch (patience 1).

        :return: Dictionary of optimizer and learning rate scheduler configuration.
        optimizer: The Adam optimizer, weight decay: 0.0001
        lr_scheduler:
            - scheduler: Defined scheduler
            - monitor: Metric to monitor (val_loss)
            - interval: Interval for updating (epoch)
            - frequency: Frequency of update (every epoch)
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
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
        logger.info(
            f"Validation epoch ended. Val loss: {val_loss.item()} | Best val loss: {self.val_loss_best.compute().item()}")
        self.val_loss_metric.reset()

    def on_test_epoch_end(self):
        avg_stoi = np.mean(self.test_stoi_values) if self.test_stoi_values else 0.0
        logger.info(f"Test epoch ended. Test loss: {self.test_loss_metric.compute().item()}")
        logger.info(f"Test Metrics - SDR: {self.test_sdr_metric.compute().item():.2f} dB, STOI: {avg_stoi:.2f}")
        self.log("test_STOI", avg_stoi)
        self.test_loss_metric.reset()
        self.test_sdr_metric.reset()
        self.test_stoi_values = []

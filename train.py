# train_av.py
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchmetrics import MeanMetric, MinMetric, SignalDistortionRatio
import logging
import os
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import glob

# Import your DataModule and new integrated model
from dataset_lightning.lightning_datamodule import DataModule
from AV_transformer_model.AV_transformer import AVTransformer
from AV_transformer_model.AV_module import AVTransformerLightningModule
import config

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_latest_checkpoint(checkpoint_dir):
    """Use latest saved checkpoint to resume training."""
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")), key=os.path.getmtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def train():
    """
    Training script with early stopping and checkpointing.
    """
    latest_checkpoint = get_latest_checkpoint(config.root_checkpoint)

    # Define dataset paths from config
    pretrain_root = config.PRETRAIN_DATA_PATH
    trainval_root = config.TRAINVAL_DATA_PATH
    test_root = config.TEST_DATA_PATH
    dns_root = config.DNS_DATA_PATH

    # Verify that required directories exist
    for path in [pretrain_root, trainval_root, test_root, dns_root]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Required directory not found: {path}")

    data_module = DataModule(
        pretrain_root=pretrain_root,
        trainval_root=trainval_root,
        test_root=test_root,
        dns_root=dns_root,
        snr_db=config.snr_db,
        transform=None,
        sample_rate=config.sample_rate,
        mode_prob=config.mode_prob,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        fixed_length=config.fixed_length,
        fixed_frames=config.fixed_frames,
        upsampled_sample_rate=config.upsampled_sample_rate
    )
    data_module.setup()

    # Instantiate the new integrated model.
    # (Adjust the hyperparameters as needed or use your config.)
    transformer_model_instance = AVTransformer(
        densetcn_options=config.densetcn_options,
        allow_size_mismatch=config.allow_size_mismatch,
        model_path=config.MODEL_PATH,
        use_boundary=config.use_boundary,
        relu_type=config.relu_type,
        num_classes=config.num_classes,
        backbone_type=config.backbone_type,
        chin=1,
        chout=1,
        hidden=48,
        depth=5,
        kernel_size=8,
        stride=4,
        padding=2,
        resample=3.2,
        growth=2,
        max_hidden=10000,
        normalize=False,  # Use dataset normalization
        glu=True,
        floor=1e-3,
        video_chin=512,
        d_hid=532,
        num_encoder_layers=3,
        num_heads=8,
        embed_dim=768,
        transformer_layers=3,
        transformer_ff_dim=532,
        max_seq_length=1024
    )

    model = AVTransformerLightningModule(net=transformer_model_instance, learning_rate=5e-5)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=config.root_checkpoint,
        save_top_k=-1,
        mode='min',
        filename='checkpoint_{epoch:02d}-{val_loss:.3f}',
        auto_insert_metric_name=True
    )

    csv_logger = CSVLogger(save_dir=config.log_folder)

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=config.gpus,
        callbacks=[early_stopping_callback, checkpoint_callback],
        log_every_n_steps=100,
        logger=csv_logger
    )

    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        trainer.fit(model, datamodule=data_module, ckpt_path=latest_checkpoint)
    else:
        print("Starting new training run:")
        trainer.fit(model, datamodule=data_module)

    print("Training complete!")
    print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")
    config.checkpoint = checkpoint_callback.best_model_path


if __name__ == "__main__":
    train()

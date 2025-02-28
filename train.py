import glob
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import config
from AV_transformer_model.AV_module import AVTransformerLightningModule
from AV_transformer_model.AV_transformer import AVTransformer
from dataset_lightning.lightning_datamodule import DataModule


def get_latest_checkpoint():
    """
    Checks for saved checkpoints.
    :return: Latest saved checkpoint
    """
    checkpoint_dir = config.ROOT_CHECKPOINT
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")), key=os.path.getmtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def train():
    """
    Training script with early stopping and checkpointing.
    Resumes training from a checkpoint if provided.
    """
    print("Initializing training setup...")

    # Setup pytorch lightning logger
    csv_logger = CSVLogger(save_dir=config.LOG_FOLDER)

    # Get latest saved checkpoint to resume training
    latest_checkpoint = get_latest_checkpoint()

    # Get dataset paths from config
    pretrain_root = config.PRETRAIN_DATA_PATH
    trainval_root = config.TRAINVAL_DATA_PATH
    test_root = config.TEST_DATA_PATH
    dns_root = config.DNS_DATA_PATH

    # Verify required directories exist
    for path in [pretrain_root, trainval_root, test_root, dns_root]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Required data directory not found: {path}")

    # Setup training data module
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
    data_module.setup('train_val')

    # Transformer init
    transformer_model_instance = AVTransformer(
        densetcn_options=config.densetcn_options,
        allow_size_mismatch=config.allow_size_mismatch,
        lip_reading_model_path=config.LR_MODEL_PATH,
        use_boundary=config.use_boundary,
        relu_type=config.relu_type,
        num_classes=config.num_classes,
        backbone_type=config.backbone_type,
        video_preprocessing_dim=512,
        embed_dim=768,
        max_seq_length=1024,
        orig_sample_rate=config.sample_rate,
        upsampled_sample_rate=config.upsampled_sample_rate
    )

    model = AVTransformerLightningModule(model=transformer_model_instance, learning_rate=5e-5)

    # Early stopping after 5 unimproved validation losses
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=config.ROOT_CHECKPOINT,
        save_top_k=-1,
        mode='min',
        filename='checkpoint_{epoch:02d}-{val_loss:.3f}',
        auto_insert_metric_name=True
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=config.gpus,
        callbacks=[early_stopping_callback, checkpoint_callback],
        log_every_n_steps=100,
        logger=csv_logger
    )

    # Start training
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

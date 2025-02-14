import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import os
from denoiser import pretrained
from dataset_lightning.lightning_datamodule import DataModule
from transformer.AV_transformer import AudioVideoTransformer
from transformer.transformer_model import TransformerModel
import config
import glob

def get_latest_checkpoint(checkpoint_dir):
    """ Use latest saved checkpoint to start training again"""
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")), key=os.path.getmtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def train():
    """
    Training script with early stopping and checkpointing.
    """

    latest_checkpoint = get_latest_checkpoint(config.root_checkpoint)

    # define dataset paths
    pretrain_root = config.PRETRAIN_DATA_PATH
    trainval_root = config.TRAINVAL_DATA_PATH
    test_root = config.TEST_DATA_PATH
    dns_root = config.DNS_DATA_PATH

    audio_model = pretrained.dns64()
    denoiser_decoder = audio_model.decoder

    # verify that directories exist
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
    )
    data_module.setup()

    transformer_model_instance = TransformerModel(
        audio_dim=1024,  # matches 'encoded_audio'
        video_dim=512,  # matches 'encoded_video'
        densetcn_options=config.densetcn_options,
        allow_size_mismatch=config.allow_size_mismatch,
        model_path=config.MODEL_PATH,
        use_boundary=config.use_boundary,
        relu_type=config.relu_type,
        num_classes=config.num_classes,
        backbone_type=config.backbone_type,
        embed_dim=768,
        nhead=8,
        num_layers=3,
        dim_feedforward=532,
        max_seq_length=1024,
        denoiser_decoder=denoiser_decoder
    )

    model = AudioVideoTransformer(model=transformer_model_instance, learning_rate=5e-5)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # name of the logged validation metric to monitor
        patience=5,  # number of epochs with no improvement before stopping
        mode='min'  # we want to minimize val_loss
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=config.root_checkpoint,
        save_top_k=-1,  # save after each epoch - incase training get interrupted
        mode='min',  # we want to minimize val_loss
        filename='checkpoint_{epoch:02d}-{val_loss:.3f}',
        auto_insert_metric_name=True
    )

    # log to data as export too full
    logger = CSVLogger(config.log_folder)
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=config.gpus,
        precision='16-mixed',
        callbacks=[early_stopping_callback, checkpoint_callback],
        log_every_n_steps=100,
        logger=logger
    )

    if latest_checkpoint:
        print(f"resume from last checkpoint: {latest_checkpoint}")
        trainer.fit(model, datamodule=data_module, ckpt_path=latest_checkpoint)
    else:
        print("new training:")
        trainer.fit(model, datamodule=data_module)

    print("Training complete!")
    print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")
    config.checkpoint = checkpoint_callback.best_model_path


if __name__ == "__main__":
    train()

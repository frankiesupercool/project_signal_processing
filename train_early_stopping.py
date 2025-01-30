import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os
from denoiser import pretrained
from dataset_lightning.lightning_datamodule import DataModule
from transformer.AV_transformer import AudioVideoTransformer
from transformer.transformer_model import TransformerModel
import config

def train():
    """
    Example script to train the AudioVideoTransformer with early stopping and checkpointing.
    Adjust parameters to suit your actual project requirements.
    """
    # Define dataset paths
    pretrain_root = config.PRETRAIN_DATA_PATH  # Path for pretraining data
    trainval_root = config.TRAINVAL_DATA_PATH  # Path for training-validation data
    test_root = config.TEST_DATA_PATH  # Path for testing data
    dns_root = config.DNS_DATA_PATH  # Path for DNS noise data

    audio_model = pretrained.dns64()
    denoiser_decoder = audio_model.decoder

    # Verify that directories exist
    for path in [pretrain_root, trainval_root, test_root, dns_root]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Required directory not found: {path}")

    # 1) Initialize your DataModule (update parameters as appropriate)
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
        seed=config.SEED,
    )
    data_module.setup()

    # 2) Create the underlying transformer model instance
    transformer_model_instance = TransformerModel(
        audio_dim=1024,         # matches your 'encoded_audio'
        video_dim=512,          # matches your 'encoded_video'
        densetcn_options=config.densetcn_options,
        allow_size_mismatch=config.allow_size_mismatch,
        model_path=config.MODEL_PATH,
        use_boundary=config.use_boundary,
        relu_type=config.relu_type,
        num_classes=config.num_classes,
        backbone_type=config.backbone_type,
        embed_dim=768,          # example
        nhead=8,                # example
        num_layers=3,           # example
        dim_feedforward=532,    # example
        max_seq_length=1024,    # adjust if needed
        denoiser_decoder=denoiser_decoder   # or your denoiser
    )

    # 3) Create your LightningModule with the model
    model = AudioVideoTransformer(model=transformer_model_instance, learning_rate=1e-5)

    # 4) Define callbacks for early stopping and saving the best checkpoint
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',   # name of the logged validation metric to monitor
        patience=5,           # number of epochs with no improvement before stopping
        mode='min'            # we want to minimize val_loss
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',        # name of the monitored metric
        dirpath=config.root_checkpoint,     # directory to save checkpoints
        filename='best-checkpoint',
        save_top_k=1,              # only save the best model
        mode='min'                 # we want to minimize val_loss
    )

    # 5) Setup trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        strategy='ddp_find_unused_parameters_true',# set to a higher number; early stopping may stop earlier
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices= config.gpus,
        #precision = '16-mixed',
        callbacks=[early_stopping_callback, checkpoint_callback],
        log_every_n_steps=100
    )

    # 6) Train the model
    trainer.fit(model, datamodule=data_module)

    # 7) Print path of the best checkpoint
    print("Training complete!")
    print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train()

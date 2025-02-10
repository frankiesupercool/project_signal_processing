import pytorch_lightning as pl
import torch
import os
from denoiser import pretrained
from dataset_lightning.lightning_datamodule import DataModule
from transformer.AV_transformer import AudioVideoTransformer
from transformer.transformer_model import TransformerModel
import config


def test():
    """
    Script to test the AudioVideoTransformer using the best checkpoint
    """

    print("Start testing setup")

    pretrain_root = config.PRETRAIN_DATA_PATH
    trainval_root = config.TRAINVAL_DATA_PATH
    test_root = config.TEST_DATA_PATH
    dns_root = config.DNS_DATA_PATH

    audio_model = pretrained.dns64()
    denoiser_decoder = audio_model.decoder

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
        fixed_frames=config.fixed_frames
    )
    # setup for test
    data_module.setup(stage="test")

    print("Data module setup done")

    # init transformer - must match as in train!
    transformer_model_instance = TransformerModel(
        audio_dim=1024,
        video_dim=512,
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

    print("Transformer init done")

    # load best checkpoint
    best_checkpoint_path = config.root_checkpoint + "/best-checkpoint.ckpt"

    # model = AudioVideoTransformer.load_from_checkpoint(
    #     checkpoint_path=best_checkpoint_path
    # )

    # pass transformer_model_instance
    model = AudioVideoTransformer.load_from_checkpoint(
        checkpoint_path=best_checkpoint_path,
        model=transformer_model_instance,
        learning_rate=1e-5  # check which lr was used
    )

    print("Checkpoint loaded to model - done")

    # setup trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=1
    )
    print("Trainer setup done")
    print("Start testing")

    # test loop
    test_results = trainer.test(model=model, datamodule=data_module)

    print("Testing complete!")
    print(f"Test Results: {test_results}")


if __name__ == "__main__":
    test()

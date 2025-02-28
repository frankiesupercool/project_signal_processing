import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
import os
import glob
from dataset_lightning.lightning_datamodule import DataModule
from AV_transformer_model.AV_transformer import AVTransformer
from AV_transformer_model.AV_module import AVTransformerLightningModule
import config
from pesq import pesq


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")),
                         key=os.path.getmtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def test():
    """
    Testing script that loads the best checkpoint, runs the test loop (with integrated metric computation),
    and saves the enhanced audio.
    """
    print("Setting up testing...")

    # Define dataset paths
    pretrain_root = config.PRETRAIN_DATA_PATH
    trainval_root = config.TRAINVAL_DATA_PATH
    test_root = config.TEST_DATA_PATH
    dns_root = config.DNS_DATA_PATH
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
    data_module.setup(stage="test")
    print("Data module setup complete.")

    # Instantiate the new integrated model (must match training configuration)
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
        normalize=False,  # use dataset normalization
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
    print("Transformer model instance initialized.")

    # Load best checkpoint (using config.checkpoint or get_latest_checkpoint)
    best_checkpoint_path = config.checkpoint
    model = AVTransformerLightningModule.load_from_checkpoint(
        checkpoint_path=best_checkpoint_path,
        net=transformer_model_instance,
        learning_rate=1e-5
    )
    model.eval()
    model.freeze()
    print("Checkpoint loaded to model.")

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=1
    )
    print("Starting testing...")
    test_results = trainer.test(model=model, datamodule=data_module)
    print("Testing complete!")
    print(f"Test Results: {test_results}")

    # Optionally, save enhanced audio from one test batch.
    test_loader = data_module.test_dataloader()
    test_iter = iter(test_loader)
    test_batch = next(test_iter)
    preprocessed_audio = test_batch['encoded_audio'].to(model.device)
    preprocessed_video = test_batch['encoded_video'].to(model.device)
    with torch.no_grad():
        clean_audio = model(preprocessed_audio, preprocessed_video)
    # Expected shape: [B, 1, time]; squeeze channel and concatenate along time.
    clean_audio_np = clean_audio.cpu().numpy()
    clean_audio_np = np.squeeze(clean_audio_np, axis=1)

    # PESQ metric
    if 'clean_speech' in test_batch:
        reference_audio = test_batch['clean_speech'].to(model.device)
        reference_audio_np = reference_audio.cpu().numpy().squeeze(1)
        pesq_scores = []
        # PESQ for each sample in the batch
        for ref, deg in zip(reference_audio_np, clean_audio_np):
            # ensure signals are in the range [-1, 1] and use wideband mode ('wb')
            score = pesq(config.sample_rate, ref, deg, 'wb')
            pesq_scores.append(score)
        avg_pesq = np.mean(pesq_scores)
        print(f"Average PESQ Score: {avg_pesq}")
    else:
        print("No clean audio, skipping PESQ computation.")

    concatenated_audio = np.concatenate(clean_audio_np, axis=-1)
    model_output_path = "clean_audio_long.wav"
    torchaudio.save(model_output_path, torch.tensor(concatenated_audio).unsqueeze(0), sample_rate=config.sample_rate)
    print(f"Enhanced clean audio saved to '{model_output_path}'")

    markdown_filename = "test_results.md"
    with open(markdown_filename, "w") as md_file:
        md_file.write("# Test Results\n\n")
        md_file.write("## Summary\n")
        md_file.write(f"- *Test Results:* {test_results}\n")
        md_file.write(f"- *PESQ Information:* {avg_pesq}\n")
        md_file.write(f"- *Enhanced Audio File:* {model_output_path}\n")
    print(f"Test results written to '{markdown_filename}'")


if __name__ == "__main__":
    test()
    
import os
import numpy as np
import torch
import torchaudio
from pesq import pesq
import pytorch_lightning as pl
import config
from AV_transformer_model.AV_module import AVTransformerLightningModule
from AV_transformer_model.AV_transformer import AVTransformer
from dataset_lightning.lightning_datamodule import DataModule


def test():
    """
    Testing script that loads the best checkpoint from config, runs the test loop with integrated metric computation,
    and saves the cleaned audio.
    """
    print("Set up testing...")

    # Load checkpoint from config
    best_checkpoint_path = config.checkpoint

    # Get dataset paths from config
    pretrain_root = config.PRETRAIN_DATA_PATH
    trainval_root = config.TRAINVAL_DATA_PATH
    test_root = config.TEST_DATA_PATH
    dns_root = config.DNS_DATA_PATH

    # Verify required directories exist
    for path in [pretrain_root, trainval_root, test_root, dns_root]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Required directory not found: {path}")

    # Setup test data module
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

    data_module.setup(stage="test")

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

    # Setup model with checkpoint, freeze and set to evaluation
    model = AVTransformerLightningModule.load_from_checkpoint(
        checkpoint_path=best_checkpoint_path,
        model=transformer_model_instance
    )
    model.eval()
    model.freeze()

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=1
    )

    print("Start testing...")
    test_results = trainer.test(model=model, datamodule=data_module)
    print(f"Test Results: {test_results}")

    print("Saving of predicted audio...")
    # Save predicted clean audio from a test batch
    test_loader = data_module.test_dataloader()
    test_iter = iter(test_loader)
    test_batch = next(test_iter)
    preprocessed_audio = test_batch['encoded_audio'].to(model.device)
    preprocessed_video = test_batch['encoded_video'].to(model.device)
    with torch.no_grad():
        clean_audio = model(preprocessed_audio, preprocessed_video)
    clean_audio = clean_audio.cpu().numpy()
    # [batch size, 1, time] - squeeze channel and concatenate the predictions along time.
    clean_audio = np.squeeze(clean_audio, axis=1)
    concatenated_audio = np.concatenate(clean_audio, axis=-1)
    model_output_path = os.path.join(config.INFERENCE_ROOT, "clean_audio_from_test_phase.wav")
    torchaudio.save(model_output_path, torch.tensor(concatenated_audio).unsqueeze(0), sample_rate=config.sample_rate)
    print(f"Predicted clean audio saved to '{model_output_path}'")

    # PESQ metric
    if 'clean_speech' in test_batch:
        reference_audio = test_batch['clean_speech'].to(model.device)
        reference_audio_np = reference_audio.cpu().numpy().squeeze(1)
        pesq_scores = []
        # PESQ for each sample in the batch
        for ref, deg in zip(reference_audio_np, clean_audio):
            # ensure signals are in the range [-1, 1] and use wideband mode ('wb')
            score = pesq(config.sample_rate, ref, deg, 'wb')
            pesq_scores.append(score)
        avg_pesq = np.mean(pesq_scores)
        print(f"Average PESQ Score: {avg_pesq}")
    else:
        print("No clean audio, skipping PESQ computation.")

    markdown_filename = "test_results.md"
    with open(markdown_filename, "w") as md_file:
        md_file.write("# Test Results\n\n")
        md_file.write("## Summary\n")
        md_file.write(f"- *Test Results:* {test_results}\n")
        md_file.write(f"- *PESQ Information:* {avg_pesq}\n")
        md_file.write(f"- *Enhanced Audio File:* {model_output_path}\n")
    print(f"Test results written to '{markdown_filename}'")

    print("Testing complete!")


if __name__ == "__main__":
    test()

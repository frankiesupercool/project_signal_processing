import torch
import torchaudio
import os
import config
from dataset_lightning.lightning_datamodule import DataModule
import numpy as np
import matplotlib.pyplot as plt

# Import the new integrated model.
from AV_transformer_model.AV_transformer import AVTransformer
from AV_transformer_model.AV_module import AVTransformerLightningModule  # your Lightning wrapper


def run_inference():
    """
    Runs inference on a test sample using the best model checkpoint and saves the output audio.
    """
    print("Initializing inference...")

    # Instantiate the new model using config parameters.
    new_model_instance = AVTransformer(
        densetcn_options=config.densetcn_options,
        allow_size_mismatch=config.allow_size_mismatch,
        backbone_type=config.backbone_type,
        use_boundary=config.use_boundary,
        relu_type=config.relu_type,
        num_classes=config.num_classes,
        model_path=config.MODEL_PATH,
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
        normalize=False,  # assume dataset normalization is already done
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

    print("New model instance initialized.")

    # Path to the best checkpoint.
    best_checkpoint_path = os.path.join(config.root_checkpoint, "checkpoint_epoch=02-val_loss=0.000.ckpt")

    # Load the Lightning module from checkpoint.
    model = AVTransformerLightningModule.load_from_checkpoint(
        checkpoint_path=best_checkpoint_path,
        net=new_model_instance,
        learning_rate=1e-5
    )

    model.eval()
    model.freeze()
    print("Model checkpoint loaded.")

    data_module = DataModule(
        pretrain_root=config.PRETRAIN_DATA_PATH,
        trainval_root=config.TRAINVAL_DATA_PATH,
        test_root=config.TEST_DATA_PATH,
        dns_root=config.DNS_DATA_PATH,
        snr_db=config.snr_db,
        transform=None,
        sample_rate=config.sample_rate,
        mode_prob=config.mode_prob,
        batch_size=2,  # Use a small batch for inference.
        num_workers=config.num_workers,
        fixed_length=config.fixed_length,
        fixed_frames=config.fixed_frames,
        upsampled_sample_rate=config.upsampled_sample_rate
    )
    data_module.setup(stage="test")

    test_loader = data_module.test_dataloader()
    test_iter = iter(test_loader)
    test_batch = next(test_iter)

    # Move test batch to the same device as the model.
    preprocessed_audio = test_batch['encoded_audio'].to(model.device)
    preprocessed_video = test_batch['encoded_video'].to(model.device)

    print("Running inference...")
    with torch.no_grad():
        clean_audio = model(preprocessed_audio, preprocessed_video)

    # clean_audio is expected to have shape [B, 1, time]
    # Convert to numpy, squeeze extra channel dimensions, and then concatenate along time.
    clean_audio_np = clean_audio.cpu().numpy()
    # Remove any singleton channel dimension (assuming shape [B, 1, time])
    clean_audio_np = np.squeeze(clean_audio_np, axis=1)  # Now shape becomes [B, time]
    # Concatenate the batch along the time axis.
    concatenated_audio = np.concatenate(clean_audio_np, axis=-1)
    # Convert back to a tensor of shape [1, total_time]
    audio_to_save = torch.tensor(concatenated_audio).unsqueeze(0)

    model_output_path = "clean_audio_long.wav"
    torchaudio.save(model_output_path, audio_to_save, sample_rate=config.sample_rate)
    print(f"Enhanced clean audio saved to '{model_output_path}'")

    # Save ground truth.
    clean_speech_np = test_batch['clean_speech'].cpu().numpy()  # shape: [B, 1, time]
    clean_speech_np = np.squeeze(clean_speech_np, axis=1)  # [B, time]
    concatenated_clean_speech = np.concatenate(clean_speech_np, axis=-1).astype(np.float32)
    clean_speech_tensor = torch.tensor(concatenated_clean_speech).unsqueeze(0)
    ground_truth_path = "ground_truth_clean_speech.wav"
    torchaudio.save(ground_truth_path, clean_speech_tensor, sample_rate=config.sample_rate)
    print(f"Ground truth clean speech saved to '{ground_truth_path}'")

    # Save preprocessed audio.
    preprocessed_audio_np = preprocessed_audio.cpu().numpy()  # shape: [B, 1, time]
    preprocessed_audio_np = np.squeeze(preprocessed_audio_np, axis=1)  # [B, time]
    concatenated_preprocessed_audio = np.concatenate(preprocessed_audio_np, axis=-1).astype(np.float32)
    preprocessed_audio_tensor = torch.tensor(concatenated_preprocessed_audio).unsqueeze(0)
    preprocessed_audio_path = "preprocessed_audio_long.wav"
    torchaudio.save(preprocessed_audio_path, preprocessed_audio_tensor, sample_rate=config.sample_rate)
    print(f"Preprocessed audio saved to '{preprocessed_audio_path}'")

    # Visualize the predicted clean audio.
    plt.figure(figsize=(10, 4))
    plt.plot(concatenated_audio, label="Predicted Clean Audio", color="blue")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("Predicted Clean Audio (Inference)")
    plt.legend()
    plt.show()

    # Visualize the ground truth clean audio.
    plt.figure(figsize=(10, 4))
    plt.plot(concatenated_clean_speech, label="Ground Truth Clean Audio", color="green")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("Ground Truth Clean Audio (Inference)")
    plt.legend()
    plt.show()

    print("Inference complete!")


if __name__ == "__main__":
    run_inference()
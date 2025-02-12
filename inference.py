import torch
import torchaudio
import os
import config
from denoiser import pretrained
from transformer.AV_transformer import AudioVideoTransformer
from transformer.transformer_model import TransformerModel
from dataset_lightning.lightning_datamodule import DataModule
import numpy as np


def run_inference():
    """
    Runs inference on a test sample using the best model checkpoint and saves the output audio
    """
    print("Initializing inference...")

    audio_model = pretrained.dns64()
    denoiser_decoder = audio_model.decoder

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

    best_checkpoint_path = os.path.join(config.root_checkpoint, "checkpoint_epoch=01-val_loss=0.076.ckpt")

    model = AudioVideoTransformer.load_from_checkpoint(
        checkpoint_path=best_checkpoint_path,
        model=transformer_model_instance,
        learning_rate=1e-3
    )

    model.eval()
    model.freeze()

    print("Model checkpoint loaded")

    data_module = DataModule(
        pretrain_root=config.PRETRAIN_DATA_PATH,
        trainval_root=config.TRAINVAL_DATA_PATH,
        test_root=config.TEST_DATA_PATH,
        dns_root=config.DNS_DATA_PATH,
        snr_db=config.snr_db,
        transform=None,
        sample_rate=config.sample_rate,
        mode_prob=config.mode_prob,
        batch_size=1,
        num_workers=config.num_workers,
        fixed_length=config.fixed_length,
        fixed_frames=config.fixed_frames,
        upsampled_sample_rate=config.upsampled_sample_rate
    )

    data_module.setup(stage="test")

    # get a batch of test data
    test_loader = data_module.test_dataloader()
    test_iter = iter(test_loader)
    test_batch = next(test_iter)

    # print("keys:", test_batch.keys())

    preprocessed_audio = test_batch['encoded_audio'].to(model.device)  # Move to correct device
    preprocessed_video = test_batch['encoded_video'].to(model.device)

    print("Running inference")

    with torch.no_grad():
        clean_audio = model(preprocessed_audio, preprocessed_video)

    # save enhanced audio
    # Convert tensor to numpy for saving
    clean_audio = clean_audio.squeeze(0).cpu().numpy()

    # Save cleaned audio
    save_path = "clean_audio.wav"
    torchaudio.save(save_path, torch.tensor(clean_audio).unsqueeze(0), sample_rate=config.sample_rate)

    print(f"Inference complete! Clean audio saved to {save_path}")

    # save ground Truth
    clean_speech = test_batch['clean_speech'].cpu().numpy()  # shape: (2, 1, 16000)
    clean_speech = np.squeeze(clean_speech, axis=1)  # remove extra dimension
    concatenated_clean_speech = np.concatenate(clean_speech, axis=-1).astype(np.float32)
    clean_speech_tensor = torch.tensor(concatenated_clean_speech)
    ground_truth_path = "ground_truth_clean_speech.wav"
    torchaudio.save(ground_truth_path, clean_speech_tensor.unsqueeze(0), sample_rate=config.sample_rate)
    print(f"Ground truth clean speech saved to '{ground_truth_path}'")

    # save preprocessed audio, todo sample rate
    # Downsample preprocessed audio to original sample rate
    preprocessed_audio_np = preprocessed_audio.cpu().numpy()
    preprocessed_audio_np = np.squeeze(preprocessed_audio_np, axis=1)  # remove extra dimension
    concatenated_preprocessed_audio = np.concatenate(preprocessed_audio_np, axis=-1).astype(np.float32)
    preprocessed_audio_tensor = torch.tensor(concatenated_preprocessed_audio)

    # Resample preprocessed audio to original sample rate
    preprocessed_audio_resampled = torchaudio.functional.resample(
        preprocessed_audio_tensor.unsqueeze(0),  # Add batch dimension
        orig_freq=config.upsampled_sample_rate,  # Original upsampled sample rate (e.g., 51.2 kHz)
        new_freq=config.sample_rate  # Target sample rate (e.g., 16 kHz)
    )

    # Save downsampled preprocessed audio
    preprocessed_audio_path = "preprocessed_audio_long.wav"
    torchaudio.save(preprocessed_audio_path, preprocessed_audio_resampled, sample_rate=config.sample_rate)
    print(f"Preprocessed audio saved to '{preprocessed_audio_path}'")



    import matplotlib.pyplot as plt

    #plt.plot(clean_audio.cpu().numpy()[0])
    plt.plot(clean_audio)
    plt.title("Predicted Clean Audio (Inference)")
    plt.show()

    # Visualize the ground truth clean audio
    plt.plot(clean_speech_tensor)
    plt.title("Ground Truth Clean Audio (Inference)")
    plt.show()

    print("Inference complete!")



if __name__ == "__main__":
    run_inference()

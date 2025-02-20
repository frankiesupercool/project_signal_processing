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

    best_checkpoint_path = os.path.join(config.root_checkpoint, "checkpoint_epoch=18-val_loss=0.060.ckpt")

    model = AudioVideoTransformer.load_from_checkpoint(
        checkpoint_path=best_checkpoint_path,
        model=transformer_model_instance,
        learning_rate=1e-5
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
        batch_size=2,  # two samples.. todo
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
    clean_audio = clean_audio.cpu().numpy()
    concatenated_audio = np.concatenate(clean_audio, axis=-1)
    model_output_path = "predicted_audio_files/clean_audio_long.wav"
    torchaudio.save(model_output_path, torch.tensor(concatenated_audio).unsqueeze(0), sample_rate=config.sample_rate)
    print(f"Enhanced clean audio saved to '{model_output_path}'")

    # save ground Truth
    clean_speech = test_batch['clean_speech'].cpu().numpy()  # shape: (2, 1, 16000)
    clean_speech = np.squeeze(clean_speech, axis=1)  # remove extra dimension
    concatenated_clean_speech = np.concatenate(clean_speech, axis=-1).astype(np.float32)
    clean_speech_tensor = torch.tensor(concatenated_clean_speech)
    ground_truth_path = "predicted_audio_files/ground_truth_clean_speech.wav"
    torchaudio.save(ground_truth_path, clean_speech_tensor.unsqueeze(0), sample_rate=config.sample_rate)
    print(f"Ground truth clean speech saved to '{ground_truth_path}'")

    # save preprocessed audio, todo sample rate
    preprocessed_audio_np = preprocessed_audio.cpu().numpy()
    preprocessed_audio_np = np.squeeze(preprocessed_audio_np, axis=1)  # remove extra dimension
    concatenated_preprocessed_audio = np.concatenate(preprocessed_audio_np, axis=-1).astype(np.float32)
    preprocessed_audio_tensor = torch.tensor(concatenated_preprocessed_audio)
    preprocessed_audio_path = "predicted_audio_files/preprocessed_audio_long.wav"
    torchaudio.save(preprocessed_audio_path, preprocessed_audio_tensor.unsqueeze(0), sample_rate=config.sample_rate)
    print(f"Preprocessed audio saved to '{preprocessed_audio_path}'")

    import matplotlib.pyplot as plt

    # Visualize the predicted clean audio and save
    plt.figure(figsize=(10, 4))
    plt.plot(concatenated_audio, label="Predicted Clean Audio", color="blue")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("Predicted Clean Audio (Inference)")
    plt.legend()
    predicted_audio_plot_path = os.path.join(config.plot_folder, "predicted_clean_audio.png")
    plt.savefig(predicted_audio_plot_path, dpi=300, bbox_inches="tight")
    print(f"Predicted clean audio plot saved to '{predicted_audio_plot_path}'")
    plt.close()

    # Visualize the ground truth clean audio and save
    plt.figure(figsize=(10, 4))
    plt.plot(concatenated_clean_speech, label="Ground Truth Clean Audio", color="green")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("Ground Truth Clean Audio (Inference)")
    plt.legend()
    ground_truth_plot_path = os.path.join(config.plot_folder, "ground_truth_clean_audio.png")
    plt.savefig(ground_truth_plot_path, dpi=300, bbox_inches="tight")
    print(f"Ground truth clean audio plot saved to '{ground_truth_plot_path}'")
    plt.close()

    print("Plots saved successfully!")

if __name__ == "__main__":
    run_inference()

import os
import matplotlib.pyplot as plt
import torch
import torchaudio
import config
from AV_transformer_model.AV_module import AVTransformerLightningModule
from AV_transformer_model.AV_transformer import AVTransformer
from dataset_lightning.lightning_datamodule import DataModule


def run_inference():
    """
    Runs inference for one test sample using a trained checkpoint and saves amplitude plots of clean speech,
    preprocessed audio and clean audio
    """

    print("Initializing inference...")

    # Load checkpoint from config
    best_checkpoint_path = config.checkpoint

    data_module = DataModule(
        pretrain_root=config.PRETRAIN_DATA_PATH,
        trainval_root=config.TRAINVAL_DATA_PATH,
        test_root=config.TEST_DATA_PATH,
        dns_root=config.DNS_DATA_PATH,
        snr_db=config.snr_db,
        sample_rate=config.sample_rate,
        mode_prob=config.mode_prob,
        batch_size=2,  # two samples, otherwise code rewrite needed
        num_workers=config.num_workers,
        fixed_length=config.fixed_length,
        fixed_frames=config.fixed_frames,
    )

    data_module.setup(stage="test")

    transformer_model_instance = AVTransformer(
        densetcn_options=config.densetcn_options,
        allow_size_mismatch=config.allow_size_mismatch,
        model_path=config.MODEL_PATH,
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

    model = AVTransformerLightningModule.load_from_checkpoint(
        checkpoint_path=best_checkpoint_path,
        model=transformer_model_instance
    )

    model.eval()
    model.freeze()

    # Get a batch of test data
    test_loader = data_module.test_dataloader()
    test_iter = iter(test_loader)
    test_batch = next(test_iter)

    preprocessed_audio = test_batch['encoded_audio'].to(model.device)
    preprocessed_video = test_batch['encoded_video'].to(model.device)

    print("Running inference...")

    with torch.no_grad():
        clean_audio = model(preprocessed_audio, preprocessed_video)

    # Use one of the two samples in batch
    clean_audio = clean_audio.cpu().numpy()[1]
    clean_speech = test_batch['clean_speech'].cpu().numpy()[1]
    preprocessed_audio = preprocessed_audio.cpu().numpy()[1]

    # Save audio files
    model_output_path = os.path.join(config.inference_root, "clean_audio.wav")
    torchaudio.save(model_output_path, torch.tensor(clean_audio), sample_rate=config.sample_rate)
    print(f"Enhanced clean audio saved to '{model_output_path}'")

    ground_truth_path = os.path.join(config.inference_root, "ground_truth_clean_speech.wav")
    torchaudio.save(ground_truth_path, torch.tensor(clean_speech), sample_rate=config.sample_rate)
    print(f"Ground truth clean speech saved to '{ground_truth_path}'")

    preprocessed_audio_path = os.path.join(config.inference_root, "preprocessed_audio.wav")
    torchaudio.save(preprocessed_audio_path, torch.tensor(preprocessed_audio), sample_rate=config.sample_rate)
    print(f"Preprocessed audio saved to '{preprocessed_audio_path}'")

    # Amplitude plots of audios
    clean_audio = clean_audio.flatten()
    clean_speech = clean_speech.flatten()
    preprocessed_audio = preprocessed_audio.flatten()

    plt.figure(figsize=(10, 4))
    plt.plot(clean_audio, label="Predicted Clean Audio", color="blue")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("Predicted Clean Audio (Inference)")
    plt.legend()
    predicted_audio_plot_path = os.path.join(config.plot_folder, "predicted_clean_audio.png")
    plt.savefig(predicted_audio_plot_path, dpi=300, bbox_inches="tight")
    print(f"Predicted clean audio plot saved to '{predicted_audio_plot_path}'")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(clean_speech, label="Ground Truth Clean Audio", color="green")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("Ground Truth Clean Audio (Inference)")
    plt.legend()
    ground_truth_plot_path = os.path.join(config.plot_folder, "ground_truth_clean_audio.png")
    plt.savefig(ground_truth_plot_path, dpi=300, bbox_inches="tight")
    print(f"Ground truth clean audio plot saved to '{ground_truth_plot_path}'")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(preprocessed_audio, label="Preprocessed Audio", color="red")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("Preprocessed Audio (Inference)")
    plt.legend()
    preprocessed_audio_plot_path = os.path.join(config.plot_folder, "preprocessed_audio.png")
    plt.savefig(preprocessed_audio_plot_path, dpi=300, bbox_inches="tight")
    print(f"Preprocessed audio plot saved to '{preprocessed_audio_plot_path}'")
    plt.close()

    print("Inference complete!")


if __name__ == "__main__":
    run_inference()

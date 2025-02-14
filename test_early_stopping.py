import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
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
    #best_checkpoint_path = config.root_checkpoint + "/best-checkpoint.ckpt"
    best_checkpoint_path = config.checkpoint

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
        devices=config.gpus,
        log_every_n_steps=1
    )
    print("Trainer setup done")
    print("Start testing")

    # test loop
    test_results = trainer.test(model=model, datamodule=data_module)

    print("Testing complete!")
    print(f"Test Results: {test_results}")

    print("Save enhanced audio")
    # fetch a test batch
    test_loader = data_module.test_dataloader()
    test_iter = iter(test_loader)
    test_batch = next(test_iter)

    # print("keys:", test_batch.keys())

    preprocessed_audio = test_batch['encoded_audio'].to(model.device)
    preprocessed_video = test_batch['encoded_video'].to(model.device)

    with torch.no_grad():
        clean_audio = model(preprocessed_audio, preprocessed_video)

    # save enhanced audio
    clean_audio = clean_audio.cpu().numpy()
    concatenated_audio = np.concatenate(clean_audio, axis=-1)
    model_output_path = "clean_audio_long.wav"
    torchaudio.save(model_output_path, torch.tensor(concatenated_audio).unsqueeze(0), sample_rate=config.sample_rate)
    print(f"Enhanced clean audio saved to '{model_output_path}'")

    # save ground Truth
    clean_speech = test_batch['clean_speech'].cpu().numpy()  # shape: (2, 1, 16000)
    clean_speech = np.squeeze(clean_speech, axis=1)  # remove extra dimension
    concatenated_clean_speech = np.concatenate(clean_speech, axis=-1).astype(np.float32)
    clean_speech_tensor = torch.tensor(concatenated_clean_speech)
    ground_truth_path = "ground_truth_clean_speech.wav"
    torchaudio.save(ground_truth_path, clean_speech_tensor.unsqueeze(0), sample_rate=config.sample_rate)
    print(f"Ground truth clean speech saved to '{ground_truth_path}'")

    # save preprocessed audio, todo sample rate
    preprocessed_audio_np = preprocessed_audio.cpu().numpy()
    preprocessed_audio_np = np.squeeze(preprocessed_audio_np, axis=1)  # remove extra dimension
    concatenated_preprocessed_audio = np.concatenate(preprocessed_audio_np, axis=-1).astype(np.float32)
    preprocessed_audio_tensor = torch.tensor(concatenated_preprocessed_audio)
    preprocessed_audio_path = "preprocessed_audio_long.wav"
    torchaudio.save(preprocessed_audio_path, preprocessed_audio_tensor.unsqueeze(0), sample_rate=config.sample_rate)
    print(f"Preprocessed audio saved to '{preprocessed_audio_path}'")


if __name__ == "__main__":
    test()

import os
from denoiser import pretrained
from dataset_lightning.lightning_datamodule import DataModule
from evaluation.evaluation import evaluate
from transformer.AV_transformer import AudioVideoTransformer
from transformer.transformer_model import TransformerModel
import config

def test_evaluation():
    """
    Script to test the evaluation of the AudioVideoTransformer. For that no training is done.
    """

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
        fixed_frames=config.fixed_frames,
        seed=config.SEED,
    )
    # setup for test
    data_module.setup(stage="test")

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

    model = AudioVideoTransformer(model=transformer_model_instance, learning_rate=1e-5)

    # 8) Run evaluation
    avg_pesq, avg_stoi, avg_sdr = evaluate(model, data_module)

    print("Evaluation complete!")
    print(f"For the test dataset the average_pesq is: {avg_pesq}")
    print(f"For the test dataset the average stoi is: {avg_stoi}")
    print(f"For the test dataset the average sdr is: {avg_sdr}")


if __name__ == "_main_":
    test_evaluation()

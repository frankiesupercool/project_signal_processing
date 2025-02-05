# config.py

# data paths
PRETRAIN_DATA_PATH = "../../../../data/datasets/LRS3/pretrain"       # Path to pretraining data
TRAINVAL_DATA_PATH = "../../../../data/datasets/LRS3/trainval"       # Path to training-validation data
TEST_DATA_PATH = "../../../../data/datasets/LRS3/test"               # Path to test data
DNS_DATA_PATH = "./../../../../data/datasets/denoiser_subset/datasets_fullband/noise_fullband" # Path to DNS noise data

# general configs
batch_size = 16
num_workers = 2

# setting sample size to 0.4s - audio up sampled to 51.2kHz 0.4*51.2k=20480
fixed_length = 51200
fixed_frames=25

# video encoding options
densetcn_options = {
        'block_config': [3, 3, 3, 3],
        'growth_rate_set': [384, 384, 384, 384],
        'reduced_size': 512,
        'kernel_size_set': [3, 5, 7],
        'dilation_size_set': [1, 2, 5],
        'squeeze_excitation': True,
        'dropout': 0.2,
    }

# video model and processing configurations
allow_size_mismatch = True
MODEL_PATH = './video_encoding/lrw_resnet18_dctcn_video_boundary.pth'
use_boundary = False
relu_type = "swish"
num_classes = 500
backbone_type = "resnet"


# audio model and processing configurations
snr_db = 10
sample_rate = 16000
mode_prob = {'speaker': 0.5, 'noise': 0.5}
upsample_factor = 3.2
upsampled_sample_rate = int(sample_rate * upsample_factor)

# trainer
gpus = [0]
max_epochs = 100

# root checkpoint save - public available checkpoints folder on sppc25
root_checkpoint = "../../../data/datasets/checkpoints/sp2025"

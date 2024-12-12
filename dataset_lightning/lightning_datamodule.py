import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataset_lightning.dataset import PreprocessingDataset  # Ensure correct import path


class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            pretrain_root,
            trainval_root,
            test_root,
            dns_root,
            densetcn_options,
            allow_size_mismatch,
            model_path,
            use_boundary,
            relu_type,
            num_classes,
            backbone_type,
            snr_db=0,
            transform=None,
            sample_rate=16000,
            mode_prob={'speaker': 0.5, 'noise': 0.5},
            batch_size=32,
            num_workers=4,
            fixed_length=64000,
            fixed_frames=100,
            seed=42,
    ):
        """
        PyTorch Lightning DataModule for PreprocessingDataset.

        Args:
            pretrain_root (str): Path to the pretrain dataset root directory.
            trainval_root (str): Path to the trainval dataset root directory.
            test_root (str): Path to the test dataset root directory.
            dns_root (str): Path to the DNS dataset root directory.
            snr_db (float, optional): Desired Signal-to-Noise Ratio in decibels. Defaults to 0.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
            sample_rate (int, optional): Desired sample rate for audio files. Defaults to 16000.
            mode_prob (dict, optional): Probability distribution for selecting mode. Defaults to {'speaker': 0.5, 'noise': 0.5}.
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 32.
            num_workers (int, optional): Number of worker processes for DataLoaders. Defaults to 4.
            fixed_length (int, optional): Fixed length in samples for audio waveforms. Defaults to 64000.
            fixed_frames (int, optional): Fixed number of frames for video sequences. Defaults to 100.
            train_val_split (tuple, optional): Ratios for train and validation splits from trainval. Defaults to (0.8, 0.2).
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        super().__init__()
        self.pretrain_root = pretrain_root
        self.trainval_root = trainval_root
        self.test_root = test_root
        self.dns_root = dns_root
        self.snr_db = snr_db
        self.transform = transform
        self.sample_rate = sample_rate
        self.mode_prob = mode_prob
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fixed_length = fixed_length
        self.fixed_frames = fixed_frames
        self.seed = seed
        # Audio options
        self.densetcn_options = densetcn_options
        self.allow_size_mismatch = allow_size_mismatch
        self.backbone_type = backbone_type
        self.use_boundary = use_boundary
        self.relu_type = relu_type
        self.num_classes = num_classes
        self.model_path = model_path

        # Placeholders for x_datasets
        self.pretrain_dataset = None
        self.trainval_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Download or prepare data if necessary.
        This method is called only from a single GPU.
        """
        # Since the dataset handles its own preparation (paired_files.txt),
        # no action is required here unless additional steps are needed.
        pass

    def setup(self, stage=None):
        """
        Set up x_datasets for different stages.

        Args:
            stage (str, optional): Stage to set up ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """
        general_config = {
            'dns_root': self.dns_root,
            'densetcn_options': self.densetcn_options,
            'allow_size_mismatch': self.allow_size_mismatch,
            'backbone_type': self.backbone_type,
            'use_boundary': self.use_boundary,
            'relu_type': self.relu_type,
            'num_classes': self.num_classes,
            'model_path': self.model_path,
            'snr_db': self.snr_db,
            'transform': self.transform,
            'sample_rate': self.sample_rate,
            'mode_prob': self.mode_prob,
            'fixed_length': self.fixed_length,
        }

        if stage == 'fit' or stage is None:
            # Instantiate the pretrain dataset (optional, based on your training strategy)

            self.pretrain_dataset = PreprocessingDataset(
                lrs3_root=self.pretrain_root,
                **general_config
            )

            # Instantiate the trainval dataset
            self.trainval_dataset = PreprocessingDataset(
                lrs3_root=self.trainval_root,
                **general_config
            )

        if stage == 'test' or stage is None:
            # Instantiate the test dataset
            self.test_dataset = PreprocessingDataset(
                lrs3_root=self.test_root,
                **general_config
            )

    def train_dataloader(self):
        return DataLoader(
            self.pretrain_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.trainval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

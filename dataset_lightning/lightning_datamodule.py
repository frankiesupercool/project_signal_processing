import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataset_lightning.dataset import PreprocessingDataset
import config


class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            pretrain_root,
            trainval_root,
            test_root,
            dns_root,
            snr_db=config.snr_db,
            sample_rate=16000,
            mode_prob=None,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            fixed_length=config.fixed_length,
            fixed_frames=config.fixed_frames,
    ):
        """
        PyTorch Lightning DataModule for PreprocessingDataset.

        Args:
            pretrain_root (str): Path to the pretrain dataset root directory.
            trainval_root (str): Path to the trainval dataset root directory.
            test_root (str): Path to the test dataset root directory.
            dns_root (str): Path to the DNS dataset root directory.
            snr_db (float, optional): Desired Signal-to-Noise Ratio in decibels. Defaults to 0.
            sample_rate (int, optional): Desired sample rate for audio files. Defaults to 16000.
            mode_prob (dict, optional): Probability distribution for selecting mode. Defaults to {'speaker': 0.5, 'noise': 0.5}.
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 32.
            num_workers (int, optional): Number of worker processes for DataLoaders. Defaults to 4.
            fixed_length (int, optional): Fixed length in samples for audio waveforms. Defaults to 64000.
            fixed_frames (int, optional): Fixed number of frames for video sequences. Defaults to 100.
        """
        super().__init__()
        if mode_prob is None:
            mode_prob = {'speaker': 0.5, 'noise': 0.5}
        self.pretrain_root = pretrain_root
        self.trainval_root = trainval_root
        self.test_root = test_root
        self.dns_root = dns_root
        self.snr_db = snr_db
        self.sample_rate = sample_rate
        self.mode_prob = mode_prob
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fixed_length = fixed_length
        self.fixed_frames = fixed_frames

        # Placeholders for datasets
        self.pretrain_dataset = None
        self.trainval_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        Set up datasets for different stages.

        Args:
            stage (str, optional): Stage to set up ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """
        general_config = {
            'dns_root': self.dns_root,
            'snr_db': self.snr_db,
            'sample_rate': self.sample_rate,
            'mode_prob': self.mode_prob,
            'fixed_length': self.fixed_length,
            'fixed_frames': self.fixed_frames
        }

        if stage == 'train_val' or stage is None:
            # Setup of train and validation dataset
            self.pretrain_dataset = PreprocessingDataset(
                lrs3_root=self.pretrain_root,
                **general_config,
                dataset_tag='pretrain'
            )

            self.trainval_dataset = PreprocessingDataset(
                lrs3_root=self.trainval_root,
                **general_config,
                dataset_tag='trainval'
            )

        if stage == 'test' or stage is None:
            # Setup of test dataset
            self.test_dataset = PreprocessingDataset(
                lrs3_root=self.test_root,
                **general_config,
                dataset_tag='test'
            )

    def train_dataloader(self):
        return DataLoader(
            self.pretrain_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.trainval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True
        )

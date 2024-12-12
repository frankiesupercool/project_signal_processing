import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import PreprocessingDataset

"""
One data set at a time
"""

class DataModule(pl.LightningDataModule):

    def __init__(
            self,
            densetcn_options,
            allow_size_mismatch,
            model_path,
            use_boundary,
            relu_type,
            num_classes,
            backbone_type,
            lrs3_root,
            dns_root,
            snr_db=0,
            transform=None,
            sample_rate=16000,
            mode_prob={'speaker': 0.5, 'noise': 0.5},
            batch_size=32,
            num_workers=4,
    ):

        super().__init__()
        self.lrs3_root = lrs3_root
        self.dns_root = dns_root
        self.snr_db = snr_db
        self.transform = transform
        self.sample_rate = sample_rate
        self.mode_prob = mode_prob
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Audio options
        self.densetcn_options = densetcn_options
        self.allow_size_mismatch = allow_size_mismatch
        self.backbone_type = backbone_type
        self.use_boundary = use_boundary
        self.relu_type = relu_type
        self.num_classes = num_classes
        self.model_path = model_path

    def setup(self, stage=None):
        # Split data into train, val, test
        # For simplicity, we'll assume all data is for training.
        # You can modify this method to split your data appropriately.
        self.train_dataset = PreprocessingDataset(
            self.lrs3_root, self.dns_root, self.densetcn_options, self.allow_size_mismatch, self.backbone_type,
            self.use_boundary, self.relu_type,self.num_classes, self.model_path, self.snr_db, self.transform,
            self.sample_rate, self.mode_prob
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )
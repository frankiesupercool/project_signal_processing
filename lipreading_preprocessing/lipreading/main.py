
import os
import time
from datetime import datetime
import logging

from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

from lipreading_preprocessing.lipreading.mixup import mixup_criterion, mixup_data
from lipreading_preprocessing.lipreading.model import Lipreading
from lipreading_preprocessing.lipreading.optim_utils import get_optimizer, CosineScheduler
from lipreading_preprocessing.lipreading.utils import AverageMeter, showLR, update_logger_batch, calculateNorm2, \
    load_json, CheckpointSaver, load_model
import torch
import numpy as np

from lipreading_preprocessing.lipreading.dataset import pad_packed_collate, MyDataset
from lipreading_preprocessing.lipreading.preprocess import NormalizeUtterance, Compose, Normalize, RandomCrop, \
    HorizontalFlip, TimeMask, CenterCrop, AddNoise



class LipreadingPreprocessing():


    def __init__(self, batch_size: int,
                 init_epoch: int,
                 workers:int,
                 label_path: str,
                 num_classes: int,
                 extract_feats: bool,
                 backbone_type: str,
                 interval: int,
                 alpha: int,
                 epochs: int,
                 config_path: str,
                 annonation_direc: str,
                 test: bool,
                 modality: str,
                 logging_dir: str,
                 training_mode: str,
                 data_dir:str = "test",):
        self.save_path = None
        self.lr = None
        self.training_mode = training_mode
        self.logging_dir = logging_dir
        self.use_boundary = None
        self.relu_type = None
        self.width_mult = None
        self.batch_size: int = batch_size
        self.init_epoch: int = init_epoch
        self.workers: int = workers
        self.label_path: str = label_path
        self.num_classes:int = num_classes
        self.extract_feats: bool = extract_feats
        self.data_dir: str = data_dir
        self.backbone_type: str = backbone_type
        self.interval: int = interval
        self.alpha: int = alpha
        self.epochs: int = epochs
        self.config_path: str = config_path
        self.annonation_direc: str = annonation_direc
        self.test: bool = test
        self.modality: str = modality

    def import_weights(self):
        pass

    def export_weights(self):
        pass

    def eval(self):
        pass

    def train(self, model, dset_loader, criterion, epoch, optimizer, logger):
        data_time = AverageMeter()
        batch_time = AverageMeter()

        lr = showLR(optimizer)

        logger.info('-' * 10)
        logger.info(f"Epoch {epoch}/{self.epochs - 1}")
        logger.info(f"Current learning rate: {lr}")

        model.train()
        running_loss = 0.
        running_corrects = 0.
        running_all = 0.

        end = time.time()
        for batch_idx, data in enumerate(dset_loader):
            if self.use_boundary:
                input, lengths, labels, boundaries = data
                boundaries = boundaries.cuda()
            else:
                input, lengths, labels = data
                boundaries = None
            # measure data loading time
            data_time.update(time.time() - end)

            # --
            input, labels_a, labels_b, lam = mixup_data(input, labels, self.alpha)
            labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

            optimizer.zero_grad()

            logits = model(input.unsqueeze(1).cuda(), lengths=lengths, boundaries=boundaries)

            loss_func = mixup_criterion(labels_a, labels_b, lam)
            loss = loss_func(criterion, logits)

            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # -- compute running performance
            _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_loss += loss.item()*input.size(0)
            running_corrects += lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(labels_b.view_as(predicted)).sum().item()
            running_all += input.size(0)
            # -- log intermediate results
            if batch_idx % self.interval == 0 or (batch_idx == len(dset_loader)-1):
                update_logger_batch(self, logger, dset_loader, batch_idx, running_loss, running_corrects, running_all, batch_time, data_time )

        return model


    def get_model_from_json(self):
        assert self.config_path.endswith('.json') and os.path.isfile(self.config_path), \
            f"'.json' config path does not exist. Path input: {self.config_path}"
        self_loaded = load_json( self.config_path)
        self.backbone_type = self_loaded['backbone_type']
        self.width_mult = self_loaded['width_mult']
        self.relu_type = self_loaded['relu_type']
        self.use_boundary = self_loaded.get("use_boundary", False)

        if self_loaded.get('tcn_num_layers', ''):
            tcn_options = { 'num_layers': self_loaded['tcn_num_layers'],
                            'kernel_size': self_loaded['tcn_kernel_size'],
                            'dropout': self_loaded['tcn_dropout'],
                            'dwpw': self_loaded['tcn_dwpw'],
                            'width_mult': self_loaded['tcn_width_mult'],
                          }
        else:
            tcn_options = {}
        if self_loaded.get('densetcn_block_config', ''):
            densetcn_options = {'block_config': self_loaded['densetcn_block_config'],
                                'growth_rate_set': self_loaded['densetcn_growth_rate_set'],
                                'reduced_size': self_loaded['densetcn_reduced_size'],
                                'kernel_size_set': self_loaded['densetcn_kernel_size_set'],
                                'dilation_size_set': self_loaded['densetcn_dilation_size_set'],
                                'squeeze_excitation': self_loaded['densetcn_se'],
                                'dropout': self_loaded['densetcn_dropout'],
                                }
        else:
            densetcn_options = {}

        model = Lipreading( modality=self.modality,
                            num_classes=self.num_classes,
                            tcn_options=tcn_options,
                            densetcn_options=densetcn_options,
                            backbone_type=self.backbone_type,
                            relu_type=self.relu_type,
                            width_mult=self.width_mult,
                            use_boundary=self.use_boundary,
                            extract_feats=self.extract_feats).cuda()
        calculateNorm2(model)
        return model

    def get_preprocessing_pipelines(self):
        # -- preprocess for the video stream
        preprocessing = {}
        # -- LRW config
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
            Normalize(0.0, 255.0),
            RandomCrop(crop_size),
            HorizontalFlip(0.5),
            Normalize(mean, std),
            TimeMask(T=0.6 * 25, n_mask=1)
        ])

        preprocessing['val'] = Compose([
            Normalize(0.0, 255.0),
            CenterCrop(crop_size),
            Normalize(mean, std)
        ])

        preprocessing['test'] = preprocessing['val']

        return preprocessing

    def get_data_loaders(self):
        preprocessing = self.get_preprocessing_pipelines()

        # create dataset object for each partition
        partitions = ['test'] if self.test else ['train', 'val', 'test']
        dsets = {partition: MyDataset(
            modality=self.modality,
            data_partition=partition,
            data_dir=self.data_dir,
            label_fp=self.label_path,
            annonation_direc=self.annonation_direc,
            preprocessing_func=preprocessing[partition],
            data_suffix='.npz',
            use_boundary=self.use_boundary,
        ) for partition in partitions}
        dset_loaders = {x: torch.utils.data.DataLoader(
            dsets[x],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=pad_packed_collate,
            pin_memory=True,
            num_workers=self.workers,
            worker_init_fn=np.random.seed(1)) for x in partitions}
        return dset_loaders

    def get_save_folder(self):
        # create save and log folder
        save_path = '{}/{}'.format(self.logging_dir, self.training_mode)
        save_path += '/' + datetime.now().isoformat().split('.')[0]
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        return save_path

    def get_logger(self, save_path):
        log_path = '{}/{}_{}_{}classes_log.txt'.format(save_path, self.training_mode, self.lr, self.num_classes)
        logger = logging.getLogger("mylog")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)
        return logger

    def evaluate(self, model, dset_loader, criterion):

        model.eval()

        running_loss = 0.
        running_corrects = 0.

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(dset_loader)):
                if self.use_boundary:
                    input, lengths, labels, boundaries = data
                    boundaries = boundaries.cuda()
                else:
                    input, lengths, labels = data
                    boundaries = None
                logits = model(input.unsqueeze(1).cuda(), lengths=lengths, boundaries=boundaries)
                _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
                running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

                loss = criterion(logits, labels.cuda())
                running_loss += loss.item() * input.size(0)

        print(f"{len(dset_loader.dataset)} in total\tCR: {running_corrects / len(dset_loader.dataset)}")
        return running_corrects / len(dset_loader.dataset), running_loss / len(dset_loader.dataset)

    def main(self):
        save_path = self.get_save_folder()
        ckpt_saver = CheckpointSaver(save_path)

        logger = self.get_logger(save_path)

        #Define model parameters form json lrw_resnet18_dctcn_boundary.json
        backbone_type = "resnet"
        width_mult = 1.0
        relu_type = "swish"
        use_boundary = True
        densetcn_options = {'block_config': [3,
                                             3,
                                             3,
                                             3],
                            'growth_rate_set': [384,
                                                384,
                                                384,
                                                384],
                            'reduced_size': 512,
                            'kernel_size_set': [3,
                                                5,
                                                7],
                            'dilation_size_set': [1,
                                                  2,
                                                  5],
                            'squeeze_excitation': True,
                            'dropout': 0.2,
                            }

        tcn_options = {}
        # Initialise Model
        model = Lipreading( modality=self.modality,
                        num_classes=self.num_classes,
                        tcn_options=tcn_options,
                        densetcn_options=densetcn_options,
                        backbone_type=backbone_type,
                        relu_type=relu_type,
                        width_mult=width_mult,
                        use_boundary=use_boundary,
                        extract_feats=self.extract_feats).cuda()



        #TODO continue refactoring and understanding from here



        dset_loaders = self.get_data_loaders()
        # -- get loss function
        criterion = nn.CrossEntropyLoss()
        # -- get optimizer
        optimizer = get_optimizer(self, optim_policies=model.parameters())
        # -- get learning rate scheduler
        scheduler = CosineScheduler(self.lr, self.epochs)

        epoch = self.init_epoch

        while epoch < self.epochs:
            model = self.train(model, dset_loaders['train'], criterion, epoch, optimizer, logger)
            acc_avg_val, loss_avg_val = self.evaluate(model, dset_loaders['val'], criterion)
            logger.info(f"{'val'} Epoch:\t{epoch:2}\tLoss val: {loss_avg_val:.4f}\tAcc val:{acc_avg_val:.4f}, LR: {showLR(optimizer)}")
            # -- save checkpoint
            save_dict = {
                'epoch_idx': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            ckpt_saver.save(save_dict, acc_avg_val)
            scheduler.adjust_lr(optimizer, epoch)
            epoch += 1

        # -- evaluate best-performing epoch on test partition
        best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.get_best_fn())
        load_model(best_fp, model)
        acc_avg_test, loss_avg_test = self.evaluate(model, dset_loaders['test'], criterion)
        logger.info(f"Test time performance of best epoch: {acc_avg_test} (loss: {loss_avg_test})")

if __name__ == '__main__':
    lipreading_preprocessing = LipreadingPreprocessing()
    lipreading_preprocessing.main()

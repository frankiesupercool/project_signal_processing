import torch

from video_encoding.model import Lipreading
from video_encoding.utils import load_model
from utils.device_utils import get_device

device = get_device()

class LipreadingPreprocessing:

    def __init__(self,
                 allow_size_mismatch: bool,
                 model_path: str,
                 use_boundary: bool,
                 relu_type: str,
                 num_classes: int,
                 backbone_type: str,
                 densetcn_options
                 ):
        self.allow_size_mismatch = allow_size_mismatch
        self.model_path = model_path
        self.use_boundary = use_boundary
        self.relu_type = relu_type
        self.num_classes = num_classes
        self.backbone_type = backbone_type
        self.densetcn_options = densetcn_options

        self.create_model()
        self.model = load_model(self.model_path, self.model, allow_size_mismatch=self.allow_size_mismatch)



    def create_model(self):

        # Define model parameters form json lrw_resnet18_dctcn_boundary.json
        backbone_type = "resnet"
        relu_type = "swish"
        use_boundary = True


        # Initialise Model
        self.model = Lipreading(num_classes=self.num_classes,
                           densetcn_options=self.densetcn_options,
                           backbone_type=backbone_type,
                           relu_type=relu_type,
                           use_boundary=use_boundary).to(device)
        print(self.model)

    def extract_feats(self, model, data):
        """
        :rtype: FloatTensor
        """
        # model.eval()
        # preprocessing_func = get_preprocessing_pipelines()['test']
        # data = preprocessing_func(np.load(args.mouth_patch_path)['data'])  # data: TxHxW

        return model(torch.FloatTensor(data)[None, None, :, :, :].to(device), lengths=[data.shape[0]])

    def generate_encodings(self, data):
        return self.extract_feats(self.model, data).to(device).detach().numpy()


'''
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
                            use_boundary=self.use_boundary).cuda()
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
        '''


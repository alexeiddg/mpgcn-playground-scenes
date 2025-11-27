import os, yaml, warnings, logging, torch, numpy as np
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis

from . import utils
from . import dataset
from . import model
from . import scheduler


class Initializer():
    def __init__(self, args):
        self.args = args
        self.init_save_dir()

        logging.info('')
        self.init_environment()
        self.init_device()
        self.init_dataloader()
        self.init_model()
        self.init_optimizer()
        self.init_lr_scheduler()
        self.init_loss_func()
        logging.info('Successful!')
        logging.info('')

    def init_save_dir(self):
        self.save_dir = utils.set_logging(self.args)
        with open('{}/config.yaml'.format(self.save_dir), 'w') as f:
            yaml.dump(vars(self.args), f)
        logging.info('Saving folder path: {}'.format(self.save_dir))

    def init_environment(self):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        self.global_step = 0
        if self.args.debug:
            self.model_name = 'debug'
            self.scalar_writer = None
        elif self.args.evaluate or self.args.extract:
            self.model_name = '{}_{}'.format(self.args.model_type, self.args.dataset)
            self.scalar_writer = None
            warnings.filterwarnings('ignore')
        else:
            self.model_name = '{}_{}'.format(self.args.model_type, self.args.dataset)
            self.scalar_writer = SummaryWriter(logdir=self.save_dir)
            warnings.filterwarnings('ignore')
        logging.info('Saving model name: {}'.format(self.model_name))

    def init_device(self):
        if type(self.args.gpus) is int:
            self.args.gpus = [self.args.gpus]
        if len(self.args.gpus) > 0 and torch.cuda.is_available():
            for i in self.args.gpus:
                mem_free, mem_total = torch.cuda.mem_get_info(i)
                if (mem_total - mem_free) // (2**20) > 8192:
                    logging.info('GPU-{} is occupied!'.format(i))
                    raise ValueError()
            self.output_device = self.args.gpus[0]
            self.device =  torch.device('cuda:{}'.format(self.output_device))
            torch.cuda.set_device(self.output_device)
        else:
            logging.info('Using CPU!')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.output_device = None
            self.device =  torch.device('cpu')

    def init_dataloader(self):
        dataset_name = self.args.dataset.split('-')[0]
        dataset_args = self.args.dataset_args
        dataset_args['debug'] = self.args.debug
        dataset_args.setdefault('augment', True)
        if self.args.debug:
            dataset_args['augment'] = False
        self.train_batch_size = dataset_args['train_batch_size']
        self.eval_batch_size = dataset_args['eval_batch_size']
        self.feeders, self.data_shape, self.num_class, self.A, self.parts = dataset.create(
            self.args.dataset, **dataset_args
        )
        train_area_ids = getattr(self.feeders['train'], 'area_ids', None)
        if train_area_ids is not None:
            self.num_areas = int(np.max(train_area_ids)) + 1
        else:
            self.num_areas = getattr(self.args, "num_areas", None) \
                or self.args.model_args.get("num_areas", 2)

        train_labels = []
        for _, y, _, _ in self.feeders['train']:
            train_labels.append(int(y))

        train_labels = np.array(train_labels)

        # Class frequencies
        observed_classes = int(train_labels.max()) + 1 if train_labels.size else self.num_class
        if observed_classes != self.num_class:
            logging.warning(
                'Adjusting num_class from {} to {} based on observed train labels.'.format(
                    self.num_class, observed_classes
                )
            )
            self.num_class = observed_classes

        class_counts = np.bincount(train_labels, minlength=self.num_class) + 1e-6
        class_weights = 1.0 / class_counts

        # Weight per sample
        sample_weights = class_weights[train_labels]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

        from torch.utils.data import WeightedRandomSampler
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        self.train_loader = DataLoader(self.feeders['train'],
            batch_size=self.train_batch_size, num_workers=4*len(self.args.gpus),
            pin_memory=True, sampler=train_sampler, shuffle=False, drop_last=True
        )
        self.eval_loader = DataLoader(self.feeders['eval'],
            batch_size=self.eval_batch_size, num_workers=4*len(self.args.gpus),
            pin_memory=True, shuffle=False, drop_last=False
        )
        logging.info('Dataset: {}'.format(self.args.dataset))
        logging.info('Batch size: train-{}, eval-{}'.format(self.train_batch_size, self.eval_batch_size))
        logging.info('Data shape (branch, channel, frame, joint, person): {}'.format(self.data_shape))
        logging.info('Number of action classes: {}'.format(self.num_class))

    def init_model(self):
        kwargs = {
            'data_shape': self.data_shape,
            'num_class': self.num_class,
            'A': torch.Tensor(self.A),
            'parts': self.parts,
            'num_areas': self.num_areas,
        }
        self.model = model.create(self.args.model_type, **(self.args.model_args), **kwargs)
        head_out = getattr(self.model, 'fcn', None)
        if head_out is not None and hasattr(head_out, 'out_features'):
            logging.info('Classifier head out_features: {}'.format(head_out.out_features))
        logging.info('Model: {} {}'.format(self.args.model_type, self.args.model_args))
        with open('{}/model.txt'.format(self.save_dir), 'w') as f:
            print(self.model, file=f)

        flops = FlopCountAnalysis(deepcopy(self.model), inputs=torch.rand([1]+self.data_shape))
        flops.unsupported_ops_warnings(False)
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info('Model profile: {:.2f}G FLOPs and {:.2f}M Parameters'.format(flops.total() / 1e9 , params / 1e6))

        self.model = torch.nn.DataParallel(
            self.model.to(self.device), device_ids=self.args.gpus, output_device=self.output_device
        )
        pretrained_model = '{}/{}.pth.tar'.format(self.args.pretrained_path, self.model_name)
        if os.path.exists(pretrained_model):
            checkpoint = torch.load(pretrained_model, map_location=torch.device('cpu'))
            self.model.module.load_state_dict(checkpoint['model'])
            self.cm = checkpoint['best_state']['cm']
            logging.info('Pretrained model: {}'.format(pretrained_model))
        elif self.args.pretrained_path:
            logging.warning('Warning: Do NOT exist this pretrained model: {}!'.format(pretrained_model))
            logging.info('Create model randomly.')

    def init_optimizer(self):
        try:
            optimizer = utils.import_class('torch.optim.{}'.format(self.args.optimizer))
        except:
            logging.warning('Warning: Do NOT exist this optimizer: {}!'.format(self.args.optimizer))
            logging.info('Try to use SGD optimizer.')
            self.args.optimizer = 'SGD'
            optimizer = utils.import_class('torch.optim.SGD')
        optimizer_args = self.args.optimizer_args[self.args.optimizer]
        self.optimizer = optimizer(self.model.parameters(), **optimizer_args)
        logging.info('Optimizer: {} {}'.format(self.args.optimizer, optimizer_args))

    def init_lr_scheduler(self):
        scheduler_args = self.args.scheduler_args[self.args.lr_scheduler]
        self.max_epoch = scheduler_args['max_epoch']
        lr_scheduler = scheduler.create(self.args.lr_scheduler, len(self.train_loader), **scheduler_args)
        self.eval_interval, lr_lambda = lr_scheduler.get_lambda()
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        logging.info('LR_Scheduler: {} {}'.format(self.args.lr_scheduler, scheduler_args))

    def init_loss_func(self):
        train_labels = []
        train_feeder = self.feeders['train']

        for _, y, _, _ in train_feeder:
            train_labels.append(int(y))

        train_labels = np.array(train_labels)

        class_counts = np.bincount(train_labels, minlength=self.num_class)
        class_counts = class_counts + 1e-6

        class_weights = 1.0 / class_counts
        class_weights = class_weights * (self.num_class / class_weights.sum())

        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        self.loss_func = torch.nn.CrossEntropyLoss(weight=weight_tensor).to(self.device)

        logging.info(
            f'Loss function: {self.loss_func.__class__.__name__} with class weights: {class_weights.tolist()}'
        )

import os
import argparse, yaml

# mini-classification basic builders
from config import get_config
from data.build import build_loader
from models.build import build_model
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from trainer import build_epoch_trainer

# pytorch-lightning
from pytorch_lightning.core.lightning import LightningModule

# pytorch and timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy

def parse_option():
    parser = argparse.ArgumentParser(description='mini-classification args', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'], help='supported dataset')
    parser.add_argument('--data-path', type=str, default='./dataset', help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--log-output', default='log-output', type=str, metavar='PATH',
                        help='root of log output folder, the full path is <log-output>/<model_name>/<tag> (default: log-output)')
    parser.add_argument('--use-checkpoint', action='store_true', help='whether to use gradient checkpointing to save memory')
    parser.add_argument('--gpu', type=str, help="gpu choose, eg. '0,1,2,...' ")
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--debug', action='store_true', help='Perform debug only')
    parser.add_argument('--tag', type=str, default='default', help='name of this training')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--precision', type=int, default=16, choices=[16, 32], help='whether to use fp16 training')
    parser.add_argument('--accelerator', type=str, default='dp', choices=['dp', 'ddp'], help='DataParallel or Distributed DataParallel')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def lightning_train_wrapper(model, criterion, optimizer, lr_scheduler, mixup_fn, mixup: bool):
    class Lightning_Training(LightningModule):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.save_hyperparameters(config)
            # self.automatic_optimization = False # close the automatic optimize
            self.model = model
            self.criterion = criterion
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
            self.mixup_fn = mixup_fn
            self.mixup = mixup

        def forward(self, x):
            x = self.model(x)
            return x

        def training_step(self, batch, batch_idx):
            samples, targets = batch
            if self.mixup:
                samples, targets = mixup_fn(samples, targets)
                outputs = self.forward(samples)
                loss = self.criterion(outputs, targets)
            else:
                outputs = self.forward(samples)
                loss = self.criterion(outputs, targets.long())
            self.lr_scheduler.step()
            self.log('training loss', loss)
            self.log('lr', self.lr_scheduler.get_lr()[0])
            return loss

        def validation_step(self, batch, batch_idx):
            samples, targets = batch
            targets = targets.long()
            outputs  = self(samples)
            loss = F.cross_entropy(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            self.log_dict({'validation loss': loss, 'acc1': acc1, 'acc5': acc5}, on_epoch=True)

        def configure_optimizers(self):
            optimizer = self.optimizer
            # scheduler = {
            #     'scheduler': self.lr_scheduler,
            #     'interval': 'step',
            # }
            return {'optimizer': optimizer}
    return Lightning_Training

def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    model = build_model(config)
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    trainer = build_epoch_trainer(config)
    mixup = True
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        # close mixup
        mixup = False
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        # close mixup
        mixup = False
        criterion = torch.nn.CrossEntropyLoss()
     
    lightning_train_engine = lightning_train_wrapper(model, criterion, optimizer, lr_scheduler, mixup_fn, mixup)
    lightning_model = lightning_train_engine(config)
    trainer.fit(
        model=lightning_model, 
        train_dataloader=data_loader_train,
        val_dataloaders=data_loader_val,
    )



if __name__ == '__main__':
    _, config = parse_option()
    main(config)
import os
import argparse, yaml

# mini-classification basic builders
from config import get_config
from data.build import build_loader
from models.build import build_model
from optimizer import build_optimizer
from lr_scheduler import build_scheduler

# pytorch-lightning
from pytorch_lightning.core.lightning import LightningModule

# pytorch and timm
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    parser.add_argument('--data-path', type=str, default='./data', help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--use-checkpoint', action='store_true', help='whether to use gradient checkpointing to save memory')
    parser.add_argument('--gpu', type=str, help="gpu choose, eg. '0,1,2,...' ")
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--tag', type=str, default='test', required=True, help='name of this training')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--precision', type=int, default=16, choices=[16, 32], help='whether to use fp16 training')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def train_engine(config, model, criterion, optimizer, lr_scheduler):
    class Lightning_Training(LightningModule):
        def __init__(self, config, hparams):
            super().__init__()
            self.config = config
            self.save_hyperparameters(hparams)
            self.model = model
            self.criterion = criterion
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

        def forward(self, x):
            x = self.model(x)
            return x

        def training_step(self, batch, batch_idx):
            images, labels = batch
            preds = self.forward(images)
            loss = self.criterion(preds, labels)
            self.log('loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            images, labels = batch
            test_outputs = self(images)
            loss = F.cross_entropy(test_outputs, labels)
            _, pred = test_outputs.topk(5, 1, largest=True, sorted=True)
            labels = labels.view(labels.size(0), -1).expand_as(pred)
            correct = pred.eq(labels).float()

            # top-5
            correct_5 = correct[:, :5].sum()

            # top-1
            correct_1 = correct[:, :1].sum()
            self.log_dict({'test_loss': loss, 'top-1': correct_1, 'top-5': correct_5}, on_epoch=True)

        def configure_optimizers(self):
            optimizer = self.optimizer
            scheduler = {
                'scheduler': self.lr_scheduler,
                'interval': 'step',
            }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    model = build_model(config)
    optimizer = build_optimizer(config)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))




if __name__ == '__main__':
    _, config = parse_option()
    print("Hyper parameters:")
    print(configs)
    # if configs.run_mode == 'train':
    #     train_engine(configs)
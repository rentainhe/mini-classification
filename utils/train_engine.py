# import all you need
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from models.get_network import get_network
from optim.get_optim import get_optim
from dataloader.get_dataloader import get_train_loader, get_test_loader
from scheduler.get_scheduler import get_scheduler
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.callbacks import save_monitor
from utils.check_config import check_config

def train_engine(__C):
    # check if there are anything wrong in the configs
    check_config(__C)

    # define training loop in LightningModule
    class Lightning_Training(LightningModule):
        def __init__(self, config, hparams):
            super().__init__()
            self.__C = config
            self.save_hyperparameters(hparams)
            self.net = get_network(config)

        def forward(self, x):
            x = self.net(x)
            return x

        def training_step(self, batch, batch_idx):
            images, labels = batch
            preds = self.forward(images)
            loss = F.cross_entropy(preds, labels)
            self.log('loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            images, labels = batch
            preds = self(images)
            loss = F.cross_entropy(preds, labels)
            labels_hat = torch.argmax(preds, dim=1)
            test_acc = torch.sum(labels == labels_hat).item() / (len(labels) * 1.0)
            self.log_dict({'test_loss': loss, 'test_acc': test_acc}, on_epoch=True)

        def configure_optimizers(self):
            optimizer = get_optim(self.__C, self.parameters())
            scheduler = {
                'scheduler': get_scheduler(self.__C, optimizer),
                'interval': 'step',
            }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    # define trainloader and testloader
    train_loader = get_train_loader(__C)
    test_loader = get_test_loader(__C)

    # define callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    save_checkpoint_monitor = save_monitor(__C)

    Lightning_Training = Lightning_Training(__C,
                                            __C.__dict__)

    # define Trainer
    trainer = Trainer(max_steps=__C.training['max_steps'],
                      gpus=__C.accelerator['gpus'],
                      accumulate_grad_batches= __C.training['gradient_accumulation_steps'],
                      callbacks=[lr_monitor, save_checkpoint_monitor],
                      precision=__C.training['precision'],
                      resume_from_checkpoint=__C.training['resume_from_checkpoint'],
                      auto_select_gpus=__C.training['auto_select_gpus'],
                      val_check_interval=__C.training['val_check_interval'],
                      accelerator=__C.accelerator['mode'])

    trainer.fit(Lightning_Training, train_loader, test_loader)

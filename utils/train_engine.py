# import all you need
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from models.get_network import get_network
from optim.get_optim import get_optim
from dataloader.get_dataloader import get_train_loader, get_test_loader
from torchvision.models import resnet18


def train_engine(__C):
    # define training loop in LightningModule

    class Lightning_Training_module(LightningModule):
        def __init__(self, __C):
            super().__init__()
            self.__C = __C
            self.net = get_network(__C)

        def forward(self, x):
            x = self.net(x)
            return x

        def training_step(self, batch, batch_idx):
            images, labels = batch
            preds = self.forward(images)
            loss = F.cross_entropy(preds, labels)
            return {'loss': loss}

        def configure_optimizers(self):
            optimizer = get_optim(self.__C, self.parameters())
            return optimizer

    # define trainloader and testloader
    train_loader = get_train_loader(__C)
    test_loader = get_test_loader(__C)

    Lightning_Training = Lightning_Training_module(__C)
    trainer = Trainer(max_epochs=5, gpus=1)
    trainer.fit(Lightning_Training, train_loader)



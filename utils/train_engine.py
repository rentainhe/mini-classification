# import all you need
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from models.get_network import get_network
from optim.get_optim import get_optim
from dataloader.get_dataloader import get_train_loader, get_test_loader

def train_engine(__C):
    model = get_network(__C)

    # define training loop in LightningModule
    class Extend_Pytorch_Lightning(model, LightningModule):
        def __init__(self, __C):
            super().__init__()
            self.__C = __C

        def training_step(self, batch, batch_idx):
            images, labels = batch
            preds = self.forward(images)
            loss = F.cross_entropy(preds, labels)
            return {'loss': loss}

        def configure_optimizers(self):
            optimizer = get_optim(self.__C)
            return optimizer

    # define trainloader and testloader
    train_loader = get_train_loader(__C)
    test_loader = get_test_loader(__C)

    Lightning_Model = Extend_Pytorch_Lightning(__C)
    trainer = Trainer(max_epochs=5)
    trainer.fit(Lightning_Model, train_loader)



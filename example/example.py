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
from pytorch_lightning.core import LightningModule
from scheduler.scheduler import WarmupCosineSchedule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import metrics

# transforms
# prepare transforms standard to MNIST
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])
mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
# mnist_train = DataLoader(mnist_train, batch_size=64)

mnist_val = MNIST(os.getcwd(), train=False, download=True, transform=transform)
mnist_val = DataLoader(mnist_val, batch_size=64)

from models.net.resnet import BasicBlock
from models.net.resnet import resnet18

net = resnet18()
# build your model
class StandardMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer1 = torch.nn.Linear(28 * 28, 128)
        self.layer2 = torch.nn.Linear(128, 256)
        self.layer3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        x = self.layer1(x)
        x = torch.relu(x)

        x = self.layer2(x)
        x = torch.relu(x)

        x = self.layer3(x)
        x = torch.log_softmax(x, dim=1)

        return x


# extend StandardMNIST and LightningModule at the same time
# this is what I like from python, extend two class at the same time
class ExtendMNIST(StandardMNIST, LightningModule):
    def __init__(self, __C=None):
        super().__init__()
        self.train_acc = metrics.Accuracy()
        self.valid_acc = metrics.Accuracy()
        self.batch_size = 64
        self.mnist_train = mnist_train
    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self.forward(data)
        loss = F.nll_loss(logits, target)
        self.train_acc(logits, target)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return {'loss': loss}

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
        optimier = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        train_sched = {
            'scheduler': WarmupCosineSchedule(optimier, warmup_steps=1000, t_total=80000),
            'interval': 'step',
            'frequency': 1
        }
        return {'optimizer': optimier, 'lr_scheduler': train_sched}

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

model = ExtendMNIST()
lr_monitor = LearningRateMonitor(logging_interval='step')

tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs/', name='metrics_test',version='metrics2')

trainer = Trainer(max_steps=90000, callbacks=[lr_monitor], logger=tb_logger, auto_scale_batch_size='power')
trainer.tune(model)
trainer.fit(model, mnist_train, mnist_val)

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

# transforms
# prepare transforms standard to MNIST
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])
mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
mnist_train = DataLoader(mnist_train, batch_size=64)

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
class ExtendMNIST(LightningModule):
    def __init__(self, __C=None):
        super().__init__()
        self.net = resnet18()

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self.forward(data)
        loss = F.nll_loss(logits, target)
        labels_hat = torch.argmax(logits, dim=1)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

model = ExtendMNIST()
trainer = Trainer(max_epochs=5)
trainer.fit(model, mnist_train)

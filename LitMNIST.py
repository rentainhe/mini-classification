import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
import os
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
import torch.optim as optim

# 定义 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])
mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
mnist_train = DataLoader(mnist_train, batch_size=64)

mnist_val = MNIST(os.getcwd(), train=False, download=True, transform=transform)
mnist_val = DataLoader(mnist_val, batch_size=64)


class LitMNIST(LightningModule):

    def __init__(self):
        super().__init__()

        # mnist images are (1,28,28)
        self.layer_1 = torch.nn.Linear(28* 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        b, c, h ,w = x.size()
        x = x.view(b,-1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    '''
        新功能：在模块内部添加 training loop
    '''
    def training_step(self, batch, batch_idx):
        x, y =batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits,y)
        self.log('val_loss', loss, on_step=True)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=1e-3)



# 除非有添加新的模块功能，不然LightningModule就完全等于nn.Module

# 有两种训练形式，第一种是在fit中标注出数据集
model = LitMNIST()
trainer = Trainer(max_epochs=10)
trainer.fit(model, mnist_train) # 这里加上 mnist_train 这个参数
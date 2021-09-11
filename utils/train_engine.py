import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from models.get_network import get_network
from optim.get_optim import get_optim
from dataloader.get_dataloader import get_train_loader, get_test_loader
from scheduler.get_scheduler import get_scheduler
from callbacks_old.callbacks import save_monitor
from callbacks_old.callbacks import interval_validation
from callbacks_old.get_callbacks import get_callbacks, get_callbacks_list
from pytorch_lightning import loggers as pl_loggers
from callbacks import build_callbacks
from optimizer import build_optimizer


def train_engine(__C):
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
            optimizer = get_optim(self.__C, self.parameters())
            scheduler = {
                'scheduler': get_scheduler(self.__C, optimizer),
                'interval': 'step',
            }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    # define trainloader and testloader
    train_loader = get_train_loader(__C)
    test_loader = get_test_loader(__C)

    # get callback list
    callbacks = get_callbacks_list(__C)
    # define logger
    tb_logger = pl_loggers.TensorBoardLogger(name=__C.model['name'], version=__C.name, save_dir='./tensorboard')

    # init training class
    Lightning_Training = Lightning_Training(__C,
                                            __C.__dict__)
    # define Trainer
    trainer = Trainer(max_steps=__C.training['max_steps'],
                      gpus=__C.accelerator['gpus'],
                      accumulate_grad_batches=__C.training['gradient_accumulation_steps'],
                      callbacks=callbacks,
                      precision=__C.training['precision'],
                      val_check_interval=__C.training['val_check_interval'],
                      accelerator=__C.accelerator['mode'],
                      logger=[tb_logger])
                    #   fast_dev_run=__C.debug)

    trainer.fit(model=Lightning_Training, 
    train_dataloader=train_loader, val_dataloaders=test_loader)

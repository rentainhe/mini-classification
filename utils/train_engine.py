# import all you need
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


def train_engine(__C):
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

        # def validation_step(self, batch, batch_idx):
        #     images, labels = batch
        #     preds = self(images)
        #     loss = F.cross_entropy(preds, labels)
        #     labels_hat = torch.argmax(preds, dim=1)
        #     top_1_acc = torch.sum(labels == labels_hat).item() / (len(labels) * 1.0)
        #     self.log_dict({'test_loss': loss, 'top-1': top_1_acc, 'top-5':}, on_epoch=True)

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

    # define callbacks
    # lr_monitor = LearningRateMonitor(logging_interval='step')
    # dir_path = os.path.join('ckpts',__C.name)
    # checkpoint_monitor = ModelCheckpoint(dirpath=dir_path, filename=str(__C.version), monitor='test_acc', save_top_k=1, save_last=True)
    # save_checkpoint_monitor = save_monitor(__C)
    # validation_monitor = interval_validation(__C)

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
                      auto_select_gpus=__C.training['auto_select_gpus'],
                      val_check_interval=__C.training['val_check_interval'],
                      accelerator=__C.accelerator['mode'],
                      logger=[tb_logger])
                    #   fast_dev_run=__C.debug)

    # trainer.fit(Lightning_Training, train_loader, test_loader)
    trainer.fit(model=Lightning_Training, 
    train_dataloader=train_loader, val_dataloaders=test_loader)

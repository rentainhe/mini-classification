from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from callbacks import build_callbacks

def build_trainer(config):
    tensorboard_logger = pl_loggers.TensorBoardLogger(name=config.MODEL.NAME, version=config.TAG, save_dir=config.LOG_OUTPUT)
    callbacks = build_callbacks(config)
    trainer = Trainer(
        max_epochs = config.TRAIN.EPOCHS,
        gpus = config.TRAIN.ACCELERATOR.GPUS,
        accumulate_grad_batches = config.TRAIN.ACCUMULATION_STEPS,
        callbacks = callbacks,
        precision = config.TRAIN.PRECISION,
        accelerator = config.TRAIN.ACCELERATOR.MODE,
        logger = [tensorboard_logger],
        fast_dev_run = config.DEBUG_MODE,
    )
    return trainer
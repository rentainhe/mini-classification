import os
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from callbacks import build_callbacks

def build_epoch_trainer(config):
    """
    train models in specific epochs
    the final log file will be stored as: 
    <log-output>/<dataset>/<model-type>/<model-name>/<tag>
    """
    log_save_dir = os.path.join(config.LOG_OUTPUT, config.DATA.DATASET, config.MODEL.TYPE)
    tensorboard_logger = pl_loggers.TensorBoardLogger(name=config.MODEL.NAME, 
                                                      version=config.TAG, 
                                                      save_dir=log_save_dir)
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

def build_finetune_trainer(config):
    """
    train models in specific steps/iterations
    the final log file will be stored as: 
    <log-output>/<dataset>/<model-type>/<model-name>/<tag>
    """
    log_save_dir = os.path.join(config.LOG_OUTPUT, config.DATA.DATASET, config.MODEL.TYPE)
    tensorboard_logger = pl_loggers.TensorBoardLogger(name=config.MODEL.NAME, 
                                                      version=config.TAG, 
                                                      save_dir=log_save_dir)
    callbacks = build_callbacks(config)
    trainer = Trainer(
        max_steps = config.TRAIN.STEPS,
        gpus = config.TRAIN.ACCELERATOR.GPUS,
        accumulate_grad_batches = config.TRAIN.ACCUMULATION_STEPS,
        callbacks = callbacks,
        precision = config.TRAIN.PRECISION,
        accelerator = config.TRAIN.ACCELERATOR.MODE,
        logger = [tensorboard_logger],
        fast_dev_run = config.DEBUG_MODE,
        val_check_interval = config.TRAIN.VAL_CHECK_INTERVAL
    )
    return trainer
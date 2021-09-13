import os
import pytorch_lightning.callbacks as callbacks
from pytorch_lightning.callbacks import Callback

def build_callbacks(config):
    callback_list = []
    if config.TRAIN.CALLBACKS.LEARNING_RATE_MONITOR.ENABLE:
        callback_list.append(
            callbacks.LearningRateMonitor(
                logging_interval = config.TRAIN.CALLBACKS.LEARNING_RATE_MONITOR.LOGGING_INTERVAL
                )
            )
    if config.TRAIN.CALLBACKS.MODEL_CHECKPOINT.ENABLE:
        callback_list.append(
            callbacks.ModelCheckpoint(
                dirpath = config.OUTPUT,
                filename = config.TRAIN.CALLBACKS.MODEL_CHECKPOINT.FILE_NAME,
                monitor = config.TRAIN.CALLBACKS.MODEL_CHECKPOINT.MONITOR,
                save_top_k = config.TRAIN.CALLBACKS.MODEL_CHECKPOINT.SAVE_TOP_K,
                mode = config.TRAIN.CALLBACKS.MODEL_CHECKPOINT.MODE
            )
        )
    return callback_list
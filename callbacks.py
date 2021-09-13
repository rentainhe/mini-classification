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

class save_monitor(Callback):
    def __init__(self, __C):
        self.__C = __C
        self.root = __C.save_dir['ckpts_dir']
        self.save_path = os.path.join(self.root, __C.version)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def on_batch_end(self, trainer, pl_module):
        if self.__C.training['save_checkpoint']:
            if trainer.global_step % self.__C.training['save_steps'] == 0:
                file_name = self.save_path + '/' + str(trainer.global_step) + '.ckpt'
                trainer.save_checkpoint(file_name)

class interval_validation(Callback):
    def __init__(self, __C):
        self.__C = __C
        self.validation_interval = __C.training['val_check_interval']

    def on_batch_end(self, trainer, pl_module):
        if self.__C.training['eval_every_epoch']:
            if trainer.global_step % self.validation_interval == 0:
                trainer.run_evaluation()


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
    # if config.TRAIN.CALLBACKS.INTERVAL_STEP_VALIDATE.ENABLE:
    #     callback_list.append(
    #         IntervalStepValidate(config)
    #     )
    return callback_list

# Run validation on specified steps
# class IntervalStepValidate(Callback):
#     def __init__(self, config):
#         self.config = config
#         self.total_steps = config.TRAIN.STEPS
#         self.validation_interval = config.TRAIN.CALLBACKS.INTERVAL_STEP_VALIDATE.INTERVAL

#     def on_batch_end(self, trainer, pl_module):
#         if self.total_steps % self.validation_interval == 0:
#             trainer.validate_step()
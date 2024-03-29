import torch
import math
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def build_finetune_scheduler(config, optimizer):
    num_steps = config.TRAIN.STEPS
    warmup_steps = config.TRAIN.WARMUP_STEPS

    lr_scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps=warmup_steps, t_total=num_steps
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        lr_scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=warmup_steps, t_total=num_steps
        )
    return lr_scheduler


def build_epoch_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    if config.TRAIN.ACCELERATOR.MODE == 'ddp':
        if config.TRAIN.ACCELERATOR.GPUS_PER_NODE > 1:
            num_steps = num_steps // config.TRAIN.ACCELERATOR.GPUS_PER_NODE
            warmup_steps = warmup_steps // config.TRAIN.ACCELERATOR.GPUS_PER_NODE

    lr_scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps=warmup_steps, t_total=num_steps
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        lr_scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=warmup_steps, t_total=num_steps
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'multi-step':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.TRAIN.LR_SCHEDULER.MULTISTONES, gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE
        )

    return lr_scheduler

class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
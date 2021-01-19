from scheduler.scheduler import WarmupLinearSchedule
from scheduler.scheduler import WarmupCosineSchedule

def get_scheduler(__C, optimizer):
    if __C.training['lr_scheduler'] == 'Cosine':
        return WarmupCosineSchedule(optimizer=optimizer, warmup_steps=__C.training['warmup_steps'], t_total=__C.training['max_steps'])
    elif __C.training['lr_scheduler'] == 'Linear':
        return WarmupLinearSchedule(optimizer=optimizer, warmup_steps=__C.training['warmup_steps'], t_total=__C.training['max_steps'])
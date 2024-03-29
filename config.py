import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Batch size for evaluation
_C.DATA.EVAL_BATCH_SIZE = 64
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Mean statistics for normalization
_C.DATA.MEAN = [0.485, 0.456, 0.406]
# Std statistics for normalization
_C.DATA.STD = [0.229, 0.224, 0.225]
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# ResNet parameters
_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.BLOCK = 'basic' # {'basic' or 'bottleneck'}
_C.MODEL.RESNET.NUM_BLOCK = [2, 2, 2, 2]

# DenseNet parameters
_C.MODEL.DENSENET = CN()
_C.MODEL.DENSENET.NUM_BLOCK = [6,12,24,16]
_C.MODEL.DENSENET.GROWTH_RATE = 32

# ViT parameters
_C.MODEL.VIT = CN()
_C.MODEL.VIT.PATCH_SIZE = 16
_C.MODEL.VIT.HIDDEN_SIZE = 768
_C.MODEL.VIT.MLP_DIM = 3072
_C.MODEL.VIT.NUM_HEADS = 12
_C.MODEL.VIT.NUM_LAYERS = 12
_C.MODEL.VIT.ATTENTION_DROP_RATE = 0.
_C.MODEL.VIT.DROPOUT_RATE = 0.1
_C.MODEL.VIT.CLASSIFIER = 'token'
_C.MODEL.VIT.REPRESENTATION_SIZE = None
_C.MODEL.VIT.ZERO_HEAD = False
_C.MODEL.VIT.PRETRAINED_WEIGHT = ''
# _C.MODEL.VIT.VIS = False

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

# Basic training hyper-parameters
# Training mode, 'epoch' for training model within specific epochs and 'steps' for training model within specific steps
_C.TRAIN.MODE = 'epoch'
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20

_C.TRAIN.STEPS = 20000
_C.TRAIN.WARMUP_STEPS = 500
_C.TRAIN.VAL_CHECK_INTERVAL = 100

_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
# AMP Training
_C.TRAIN.PRECISION = 16

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
# LR scheduler update frequency
_C.TRAIN.LR_SCHEDULER.FREQUENCY = 'step'
# LR scheduler type: cosine, linear, multi-step
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler, use int or list 
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# Multistones for MultiStepLR
_C.TRAIN.LR_SCHEDULER.MULTISTONES = [150, 225]

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# Callbacks
_C.TRAIN.CALLBACKS = CN()
# Monitor learning rate during training
_C.TRAIN.CALLBACKS.LEARNING_RATE_MONITOR = CN()
_C.TRAIN.CALLBACKS.LEARNING_RATE_MONITOR.ENABLE = True
_C.TRAIN.CALLBACKS.LEARNING_RATE_MONITOR.LOGGING_INTERVAL = 'step'
# Save checkpoint based on acc1
_C.TRAIN.CALLBACKS.MODEL_CHECKPOINT = CN()
_C.TRAIN.CALLBACKS.MODEL_CHECKPOINT.ENABLE = True
_C.TRAIN.CALLBACKS.MODEL_CHECKPOINT.FILE_NAME = 'best-model'
_C.TRAIN.CALLBACKS.MODEL_CHECKPOINT.MONITOR = 'acc1'
_C.TRAIN.CALLBACKS.MODEL_CHECKPOINT.SAVE_TOP_K = 1
_C.TRAIN.CALLBACKS.MODEL_CHECKPOINT.MODE = 'max'
# Validate on specified steps
# _C.TRAIN.CALLBACKS.INTERVAL_STEP_VALIDATE = CN()
# _C.TRAIN.CALLBACKS.INTERVAL_STEP_VALIDATE.ENABLE = False
# _C.TRAIN.CALLBACKS.INTERVAL_STEP_VALIDATE.INTERVAL = 1000

# Accelerator
_C.TRAIN.ACCELERATOR = CN()
_C.TRAIN.ACCELERATOR.GPUS = "1"
_C.TRAIN.ACCELERATOR.MODE = 'ddp'
_C.TRAIN.ACCELERATOR.NUM_NODES = 1
_C.TRAIN.ACCELERATOR.GPUS_PER_NODE = 1

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Range of size of the origin size cropped, used in RandomResizedCropAndInterpolation
_C.AUG.SCALE = (0.08, 1)
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'
# Probability of horizontal flip the image
_C.AUG.RANDOM_HORIZONTAL_FLIP = 0.5
# Probability of vertical flip the image
_C.AUG.RANDOM_VERTICAL_FLIP = 0.
# Random Rotation
_C.AUG.RANDOM_ROTATION = 15


# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Path to training log output folder, overwritten by comman line argument
_C.LOG_OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# Perform debug mode, quick run one step of train and val, overwritten by command line argument
_C.DEBUG_MODE = False
# local rank for DistributedDataParallel, given by command line argument
# _C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.gpu:
        config.TRAIN.ACCELERATOR.GPUS = args.gpu
        config.TRAIN.ACCELERATOR.GPUS_PER_NODE = gpu_nums = len(args.gpu.split(','))
        # scale steps due to ddp mode
        if config.TRAIN.ACCELERATOR.MODE == 'ddp':
            if config.TRAIN.ACCELERATOR.GPUS_PER_NODE > 1:
                config.TRAIN.STEPS = config.TRAIN.STEPS // gpu_nums
                config.TRAIN.WARMUP_STEPS = config.TRAIN.WARMUP_STEPS // gpu_nums

    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
        config.DATA.BATCH_SIZE = config.DATA.BATCH_SIZE // args.accumulation_steps
    if args.precision:
        config.TRAIN.PRECISION = args.precision
    if args.accelerator:
        config.TRAIN.ACCELERATOR.MODE = args.accelerator
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.output:
        config.OUTPUT = args.output
    if args.log_output:
        config.LOG_OUTPUT = args.log_output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.debug:
        config.DEBUG_MODE = True
    # if args.throughput:
    #     config.THROUGHPUT_MODE = True

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import datasets, transforms
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp

# ImageNet Default Settings
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

# Cifar100 Default Settings
CIFAR100_DEFAULT_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
CIFAR100_DEFAULT_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"successfully build val dataset")


    # TODO: add sampler
    # sampler_train = RandomSampler(dataset_train)
    # sampler_val = SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATA.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    if config.DATA.DATASET == 'imagenet':
        transform = build_transform(is_train, config)
    elif config.DATA.DATASET == 'cifar100':
        transform = build_cifar100_transform(is_train, config)

    if config.DATA.DATASET == 'cifar100':
        root = config.DATA.DATA_PATH
        dataset = datasets.CIFAR100(root=root, train=is_train, download=True, transform=transform)
        nb_classes = 100
    elif config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    else:
        raise NotImplementedError("We only support CIFAR100, ImageNet Now.")

    return dataset, nb_classes

def build_cifar100_transform(is_train, config):
    t = []
    if is_train:
        t.append(transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4))
        t.append(transforms.RandomHorizontalFlip(config.AUG.RANDOM_HORIZONTAL_FLIP))
        t.append(transforms.RandomRotation(config.AUG.RANDOM_ROTATION))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(config.DATA.MEAN, config.DATA.STD))
    return transforms.Compose(t)

def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            hflip=config.AUG.RANDOM_HORIZONTAL_FLIP,
            vflip=config.AUG.RANDOM_VERTICAL_FLIP,
            scale=config.AUG.SCALE,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(config.DATA.MEAN, config.DATA.STD))
    return transforms.Compose(t)
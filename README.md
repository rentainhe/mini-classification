# mini-classification
A simple `classification` project based on [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

## Installation
```bash
$ pip install pytorch-lightning
```

## Content
- control the training through [one config file](https://github.com/rentainhe/mini-classification/blob/master/configs/cifar100.yaml)
- pytorch-lightning [notes]() sharing
- [config file documents]()

## Usage
### 1. training
```bash
$ python run.py --run train --config cifar100 --name test
```

## Acknowledgements
Thanks a lot for these repos:
- [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)
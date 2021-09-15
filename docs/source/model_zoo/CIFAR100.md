# Model Zoo

## Environment
We use the following environment to run all the experiments on `CIFAR100` in this page.
- Python 3.6
- Pytorch 1.8.0
- Pytorch-Lightning 1.4.5
- CUDA 11.2
- CUDNN 7.6.5

## Results
We use the same hyperparameters to train all the following networks on `CIFAR100`, some networks might not get the best result under such settings. You can finetune the hyperparameters to get better results.

### Hyper-Parameters
```
DATA:
  DATASET: cifar100
  IMG_SIZE: 32
TRAIN:
  EPOCHS: 300
  WARMUP_EPOCHS: 20
  BASE_LR: 0.1
  WEIGHT_DECAY: 5e-4
  LR_SCHEDULER:
    NAME: cosine
  OPTIMIZER:
    NAME: sgd
AUG:
  MIXUP: 0.
  RANDOM_HORIZONTAL_FLIP: 0.5
  RANDOM_ROTATION: 15
MODEL:
  LABEL_SMOOTHING: 0.
```
We trained all these models using `4 GPU` with `Distributed Data-Parallel Training`. You can try to get better results by modifying these hyperparameters in each model's specific config file.

### Model List

**ResNet**
- [**Relative Paper**](https://arxiv.org/abs/1512.03385v1) 
- [**Training Log**](https://tensorboard.dev/experiment/2yn8p4FwRs2ENzZXmYKO1g/#scalars)

|   Model   | Params   | Warmup Epochs | Epochs | Optimizer | Base Lr | Scheduler | Acc1  | Acc5  |
|:----------|:--------:|:-------------:|:------:|:---------:|:-------:|:---------:|:-----:|:-----:|
| ResNet18  | 11.2M    | 20            | 300    | SGD       |  0.1    | Cosine    | 75.46 | 91.72 |
| ResNet34  | 21.3M    | 20            | 300    | SGD       |  0.1    | Cosine    | 76.15 | 92.67 |
| ResNet50  | 23.7M    | 20            | 300    | SGD       |  0.1    | Cosine    | 77.36 | 93.57 |
| ResNet101 | 42.7M    | 20            | 300    | SGD       |  0.1    | Cosine    | 78.56 | 94.13 |
| ResNet152 | 58.3M    | 20            | 300    | SGD       |  0.1    | Cosine    | 78.29 | 94.13 |

**DenseNet**
- [**Relative Paper**](https://arxiv.org/abs/1608.06993v5)
- [**Training Log**]()

|    Model     | Params   | Warmup Epochs | Epochs | Optimizer | Base Lr | Scheduler | Acc1  | Acc5  |
|:-------------|:--------:|:-------------:|:------:|:---------:|:-------:|:---------:|:-----:|:-----:|
| DenseNet121  | 7.0M     | 20            | 300    | SGD       | 0.1     | Cosine    | 78.60 | 93.75 |
| DenseNet169  | 12.6M    | 20            | 300    | SGD       | 0.1     | Cosine    | 78.36 | 93.87 |
| DenseNet201  | 18.3M    | 20            | 300    | SGD       | 0.1     | Cosine    | - | - |
| DenseNet161  | -    | 20            | 300    | SGD       | 0.1     | Cosine    | - | - |
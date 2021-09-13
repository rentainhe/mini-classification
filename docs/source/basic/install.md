# Installation
This page provides basic prerequisites to run mini-classification, including setups of datasets

## Environment Setup
**Clone this repo**
```bash
git clone https://github.com/rentainhe/mini-classification
```
**Create a conda virtual environment and activate it**
```bash
conda create -n mini python=3.7 -y
conda activate mini
```
**Install `CUDA==10.1` with `cudnn7` following the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)**

**Install `Pytorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`**
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```
**Install `Pytorch-Lightning`**
```bash
pip install pytorch-lightning
```
**Install others**
```bash
pip install -r requirements.txt
```


## Supported Datasets
- CIFAR100
- ImageNet2012

## Dataset Setup
The following datasets should be prepared before running the experiments.

**CIFAR100**

We use `torchvision.datasets.CIFAR100` directly, you only need to focus on the data dirpath which is `./dataset` by default and can be overwritten by command line with `--data-path='path/to/CIFAR100'`

**ImageNet2012**

We use standard ImageNet dataset, you can download it from http://image-net.org/. The directory struture is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/vision/stable/datasets.html?highlight=imagefolder#torchvision.datasets.ImageFolder), and the training and validation data is excepted to be in the `train/` folder and `val/` folder respectively:
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```
**Notes that please remember to set the data dirpath in training scripts with   `--data-path='path/to/imagenet/'`**
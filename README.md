# Mini-Classification
Mini-Classification是一个基于pytorch-lightning写的轻量化的图像分类代码框架


### 初衷
为什么需要基于pytorch-lightning写一个分类任务的框架？
- **学习目的** 首先pytorch-lightning是基于pytorch代码进一步wrap的一个代码框架，提供了更高级别抽象的API，让我们在使用pytorch的过程中，可以更加专注于模型部分，而不需要去考虑一些额外的代码结构，例如**分布式训练**，**Log的保存**， **resume training**， **半精度训练**等，这些在pytorch-lightning中都可以通过简单的配置实现，并且pytorch-lightning中配置了许多好用的API以及function，这让我出于学习的好奇心，想亲自体验一下pytorch-lightning的方便，并且想把方便带给他人。
- **任务简单** 分类任务是深度学习中最经典的，也是最好入门，最友好的，并且分类任务的实现并不复杂，可以让我在更新开源项目的同时，更加专注pytorch-lightning框架本身的学习。
- **适用多样性** 我思考过一个优秀的开源项目应该做到哪些部分来适用不同的人群，如果只是基于pytorch去写一个这样的框架，那其实是一件意义不大的事情，因为优秀的开源项目太多了，但是基于pytorch-lightning项目去写一个开源框架，有以下几个好处：
  - **面向小白** 对于深度学习入门选手，可以快速上手体验，方便使用者专注于模型的理解，并体验不同的模型实验效果，但是不推荐依赖于pytorch-lightning去写自己之后的代码，因为pytorch本身的使用场景还是更广，并且更有助于你对于代码内部逻辑的理解。
  - **面向进阶者** 对于进阶者而言，Mini-Classification致力于尽量将使用者研究时需要的内容容纳进来，并且将代码拓展性尽量做好，便于使用。
- **开源精神** 想做好一个完整的开源项目，如果使用者有更好的想法，非常欢迎对这个小小的开源项目提供支持和帮助。

### 基本功能
- [x] 配置统一的config文件
- [x] 实现完整的模型训练流程，并且默认保存模型训练过程中的最好结果
- [x] 配置tensorboard，记录训练过程中的变量：`training loss`, `learning rate`, `acc1`, `acc5`, `validation loss`等
- [x] 增加`step`, `cosine`, `linear`三种`lr_scheduler`。
- [x] 增加`DP`和`DDP`训练功能，并且可以通过`args`来指定特定的`device`
- [x] 增加`fp16`半精度训练

### 更新日志
如果需要了解`mini-classification`最新的更新，可以查看[changelog.md](/changelog.md)获取更多细节

## 安装与使用
### 安装
安装`pytorch-lightning`, `timm`以及其他相关库
```bash
$ pip install pytorch-lightning
$ pip install timm
$ pip install -r requirements.txt
```

### 使用
#### 1. 训练
**example**

在cifar100任务上使用指定GPU与DDP训练resnet18模型
```bash
$ python run.py --cfg configs/cifar100/resnet/resnet18.yaml --gpu 0,1 --accelerator ddp
```

## Acknowledgements
非常感谢以下优秀的开源代码所提供的参考
- [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)
- [deit](https://github.com/facebookresearch/deit)
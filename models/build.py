# --------------------------------------------------------
# mini-classification model builder
# Written by rentainhe
# --------------------------------------------------------

from models.cifar.resnet import ResNet


def build_model(config):
    dataset = config.DATA.DATASET
    model_type = config.MODEL.TYPE
    model_name = config.MODEL.NAME
    if dataset in ['cifar10', 'cifar100']:
        if model_type == 'resnet':
            from models.cifar.resnet import build_resnet
            return build_resnet(
                block_type = config.MODEL.RESNET.BLOCK,
                num_block = config.MODEL.RESNET.NUM_BLOCK
            )  
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    

        
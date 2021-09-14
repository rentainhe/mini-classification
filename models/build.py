# --------------------------------------------------------
# mini-classification model builder
# Written by rentainhe
# --------------------------------------------------------

def build_model(config):
    dataset = config.DATA.DATASET
    model_type = config.MODEL.TYPE
    model_name = config.MODEL.NAME
    if dataset == 'cifar100':
        if model_type == 'resnet':
            from models.cifar.resnet import build_resnet
            return build_resnet(
                block_type = config.MODEL.RESNET.BLOCK,
                num_block = config.MODEL.RESNET.NUM_BLOCK
            )
        elif model_type == 'densenet':
            from models.cifar.densenet import build_densenet
            return build_densenet(
                num_block = config.MODEL.DENSENET.NUM_BLOCK,
                growth_rate = config.MODEL.DENSENET.GROWTH_RATE
            )  
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    

        
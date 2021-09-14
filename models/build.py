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
    elif dataset == 'imagenet':
        if model_type == 'vit':
            from models.imagenet.vit import build_vit
            return build_vit(
                classifier = config.MODEL.VIT.CLASSIFIER,
                img_size=config.DATA.IMG_SIZE,
                patch_size=config.MODEL.VIT.PATCH_SIZE,
                num_layers=config.MODEL.VIT.NUM_LAYERS,
                hidden_size=config.MODEL.VIT.HIDDEN_SIZE,
                mlp_dim=config.MODEL.VIT.MLP_DIM,
                dropout_rate=config.MODEL.VIT.DROPOUT_RATE,
                attention_dropout_rate=config.MODEL.VIT.ATTENTION_DROP_RATE,
                num_classes=config.MODEL.NUM_CLASSES,
                zero_head=config.MODEL.VIT.ZERO_HEAD
            )  
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    

        
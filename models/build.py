# --------------------------------------------------------
# mini-classification model builder
# Written by rentainhe
# --------------------------------------------------------

def build_model(config):
    dataset = config.DATA.DATASET
    model_type = config.MODEL.TYPE
    if dataset in ['cifar10', 'cifar100']:
        if model_type == 'resnet18':
            from cifar.resnet import resnet18
            return resnet18()
        elif model_type == 'resnet34':
            from cifar.resnet import resnet34
            return resnet34()
        elif model_type == 'resnet50':
            from cifar.resnet import resnet50
            model = resnet50()
        elif model_type == 'resnet101':
            from cifar.resnet import resnet101
            return resnet101()
        elif model_type == 'resnet152':
            from cifar.resnet import resnet152
            return resnet152()

    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    

        
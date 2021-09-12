# --------------------------------------------------------
# mini-classification model builder
# Written by rentainhe
# --------------------------------------------------------

def build_model(config):
    dataset = config.DATA.DATASET
    model_type = config.MODEL.TYPE
    model_name = config.MODEL.NAME
    if dataset in ['cifar10', 'cifar100']:
        if model_type == 'resnet':
            if model_name == 'resnet18':
                from models.cifar.resnet import resnet18
                return resnet18()
            elif model_name == 'resnet34':
                from models.cifar.resnet import resnet34
                return resnet34()
            elif model_name == 'resnet50':
                from models.cifar.resnet import resnet50
                return resnet50()
            elif model_name == 'resnet101':
                from models.cifar.resnet import resnet101
                return resnet101()
            elif model_name == 'resnet152':
                from models.cifar.resnet import resnet152
                return resnet152()  

    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    

        
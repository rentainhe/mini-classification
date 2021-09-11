import torchvision.transforms as transforms


# import argparse, yaml
# from configs.build_config import configs
#
# with open('D:\Github\pytorch_lightning_learning\configs\cifar100.yaml', 'r') as f:
#     yaml_dict = yaml.load(f)
# configs.add_args(yaml_dict)
#
# # for trans in configs.transforms['img']:
# #     print(trans)
# #     print(configs.transforms['img'][trans])
#
# trans = getattr(transforms, 'Resize')
#

def get_transform_block(method, args):
    trans = getattr(transforms, method)
    enable = args['enable']
    keys = list(args)[1:]
    if enable:
        method = 'transforms.' + str(method)
        eval_str = ''
        for key in keys:
            eval_str += key + '=' + str(args[key]) + ', '
        trans = eval(method + '(' + eval_str + ')')
        return [trans]
    else:
        return []


def get_transforms(__C):
    # training transform
    train_transform_list = []
    for trans in __C.transforms['img']['train']:
        train_transform_list += get_transform_block(trans, __C.transforms['img']['train'][trans])
    train_transform_list.append(transforms.ToTensor())
    train_transform_list.append(transforms.Normalize(mean=__C.transforms['tensor']['normalization']['mean'],
                                                     std=__C.transforms['tensor']['normalization']['std']))
    train_transform = transforms.Compose(train_transform_list)

    # test transform
    test_transform_list = []
    for trans in __C.transforms['img']['test']:
        test_transform_list += get_transform_block(trans, __C.transforms['img']['test'][trans])
    test_transform_list.append(transforms.ToTensor())
    test_transform_list.append(transforms.Normalize(mean=__C.transforms['tensor']['normalization']['mean'],
                                                    std=__C.transforms['tensor']['normalization']['std']))
    test_transform = transforms.Compose(test_transform_list)

    return train_transform, test_transform

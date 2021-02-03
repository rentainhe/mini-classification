import pytorch_lightning.callbacks as callbacks
import argparse, yaml
from configs.build_config import configs

def get_callbacks(method, args):
    keys = list(args)
    method = 'callbacks.' + str(method)
    eval_str = ''
    for key in keys:
        if isinstance(args[key], str):
            eval_str += key + '=' + "'" +  args[key] + "'"+ ', '
        else:
            eval_str += key + '=' + str(args[key])  + ', '
    callback = eval(method + '(' + eval_str + ')')
    return [callback]

def get_callbacks_list(__C):
    # callback list
    callbacks_list = []
    for callback in __C.callbacks:
        if callback == 'ModelCheckpoint':
            dirpath = './ckpts/' + __C.name
            __C.callbacks[callback]['dirpath'] = dirpath
        callbacks_list += get_callbacks(callback, __C.callbacks[callback])
    return callbacks_list


# Test Code
# with open('..\configs\cifar100.yaml', 'r') as f:
#     yaml_dict = yaml.load(f)
#
# configs.add_args(yaml_dict)
#
#
# print(get_callbacks('ModelCheckpoint', configs.callbacks['ModelCheckpoint']))
# print(get_callbacks_list(configs))
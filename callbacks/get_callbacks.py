import pytorch_lightning.callbacks as callbacks

import argparse, yaml
from configs.build_config import configs

with open('D:\Github\pytorch_lightning_learning\configs\cifar100.yaml', 'r') as f:
    yaml_dict = yaml.load(f)
configs.add_args(yaml_dict)


trans = getattr(callbacks, 'LearningRateMonitor')
trans(logging_interval='step')



def get_callbacks(method, args):
    keys = list(args)
    method = 'callbacks.' + str(method)
    eval_str = ''
    for key in keys:
        if isinstance(args[key], str):
            eval_str += key + '=' + "'" +  args[key] + "'"+ ', '
        else:
            eval_str += key + '=' + args[key]  + ', '
    print(eval_str)
    callback = eval(method + '(' + eval_str + ')')
    return [callback]

get_callbacks('LearningRateMonitor', configs.callbacks['LearningRateMonitor'])
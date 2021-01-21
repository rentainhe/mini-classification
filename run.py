import argparse, yaml
from configs.build_config import configs
from utils.train_engine import train_engine

def parse_args():
    parser = argparse.ArgumentParser(description='mini-classification args')
    parser.add_argument('--config', type=str, choices=['cifar10','cifar100','imagenet'], required=True, help='choose the dataset')
    parser.add_argument('--gpu', type=str, help="gpu choose, eg. '0,1,2,...' ")
    parser.add_argument('--run', dest="run_mode", type=str, choices=['train','test'])
    parser.add_argument('--seed', type=int, help='fix random seed')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg_file = "configs/{}.yaml".format(args.config)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f)

    args_dict = configs.parse_to_dict(args)
    args_dict = {**yaml_dict, **args_dict}
    configs.add_args(args_dict)

    configs.training_init()
    configs.path_init()

    print("Hyper parameters:")
    print(configs.__dict__)
    if configs.run_mode == 'train':
        train_engine(configs)

import argparse, yaml
from configs.build_config import configs
from utils.train_engine import train_engine
from config import get_config

def parse_option():
    parser = argparse.ArgumentParser(description='mini-classification args', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default='./data', help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--use-checkpoint', action='store_true', help='whether to use gradient checkpointing to save memory')
    parser.add_argument('--gpu', type=str, help="gpu choose, eg. '0,1,2,...' ")
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--tag', type=str, default='test', required=True, help='name of this training')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--precision', type=int, default=16, choices=[16, 32], help='whether to use fp16 training')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

if __name__ == '__main__':
    _, config = parse_option()
    print(config)

    # args = parse_args()
    # cfg_file = "configs/{}.yaml".format(args.config)
    # with open(cfg_file, 'r') as f:
    #     yaml_dict = yaml.load(f)

    # args_dict = configs.parse_to_dict(args)
    # args_dict = {**yaml_dict, **args_dict}
    # configs.add_args(args_dict)

    # configs.training_init()
    # configs.path_init()

    # print("Hyper parameters:")
    # print(configs.__dict__)
    # if configs.run_mode == 'train':
    #     train_engine(configs)
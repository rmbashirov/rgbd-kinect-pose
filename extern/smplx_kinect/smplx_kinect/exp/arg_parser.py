import argparse
import yaml
from copy import deepcopy


def parse_yaml(fp):
    with open(fp, 'r') as f:
        config = yaml.safe_load(f)
    result = deepcopy(config)

    return result


def parse_args_yaml():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--train_dp', required=True)
    parser.add_argument('--fast_features_save_part', required=False)
    args = parser.parse_args()

    result = parse_yaml(args.config)
    if args.fast_features_save_part is not None:
        result['fast_features']['save']['part'] = args.fast_features_save_part
    result['train_dp'] = args.train_dp

    return result, args.config

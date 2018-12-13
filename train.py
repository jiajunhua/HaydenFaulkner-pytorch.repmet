"""
The main training script
"""
import argparse
from config.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Training Script')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    return args


def train():
    # Load Configuration

    # Load Data

    # Load Model

    # Load Optimiser

    #
    pass
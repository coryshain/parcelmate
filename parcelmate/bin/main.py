import numpy as np
import torch
import argparse

from parcelmate.cfg import get_cfg
from parcelmate.data import get_input_data
from parcelmate.model import run_connectivity

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Main executable for parcelmate package.''')
    argparser.add_argument('config', help='Path to config file.')
    args = argparser.parse_args()

    cfg = get_cfg(args.config)
    run_connectivity(
        output_dir=cfg.get('output_dir', 'results'),
        **cfg.get('connectivity', {})
    )


import argparse

from parcelmate.cfg import get_cfg
from parcelmate.model import *
from parcelmate.plot import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Main executable for parcelmate package.''')
    argparser.add_argument('config', help='Path to config file.')
    argparser.add_argument('-O', '--overwrite', action='store_true',
                           help='Recompute all outputs, even if they already exist.')
    args = argparser.parse_args()
    config = args.config
    overwrite = args.overwrite

    cfg = get_cfg(config)

    run_connectivity(
        output_dir=cfg.get('output_dir', 'results'),
        overwrite=overwrite,
        **cfg.get('connectivity', {})
    )

    run_parcellation(
        output_dir=cfg.get('output_dir', 'results'),
        overwrite=overwrite,
        **cfg.get('parcellation', {})
    )

    plot_connectivity(
        output_dir=cfg.get('output_dir', 'results')
    )

    plot_parcellation(
        output_dir=cfg.get('output_dir', 'results')
    )

    plot_stability(
        output_dir=cfg.get('output_dir', 'results')
    )


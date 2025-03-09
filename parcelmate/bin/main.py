import argparse

from parcelmate.cfg import get_cfg
from parcelmate.model import *
from parcelmate.plot import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Main executable for parcelmate package.''')
    argparser.add_argument('config_path', nargs='?', default=None, help='Path to config file.')
    argparser.add_argument('-s', '--steps', nargs='+', default=['all'], help=
                           'Space-delimited list of steps to run, or `all`.'
                           )
    argparser.add_argument('-O', '--overwrite', action='store_true',
                           help='Recompute all outputs, even if they already exist.')
    args = argparser.parse_args()
    config_path = args.config_path
    steps = set(args.steps)
    overwrite = args.overwrite

    if config_path is not None:
        cfg = get_cfg(config_path)
    else:
        cfg = {}

    if 'all' in steps or 'connectivity' in steps:
        print("Running Connectivity...\n")
        run_connectivity(
            output_dir=cfg.get('output_dir', OUTPUT_DIR),
            overwrite=overwrite,
            **cfg.get('connectivity', {})
        )

    if 'all' in steps or 'parcellation' in steps:
        print("Running Parcellation...\n")
        run_parcellation(
            output_dir=cfg.get('output_dir', OUTPUT_DIR),
            overwrite=overwrite,
            **cfg.get('parcellation', {})
        )

    if 'all' in steps or 'subnetwork_extraction' in steps:
        print("Running Subnetwork Extraction...\n")
        run_subnetwork_extraction(
            output_dir=cfg.get('output_dir', OUTPUT_DIR),
            **cfg.get('subnetwork_extraction', {})
        )

    if 'all' in steps or 'plot_connectivity' in steps:
        print("Plotting Connectivity...\n")
        plot_connectivity(
            output_dir=cfg.get('output_dir', OUTPUT_DIR)
        )

    if 'all' in steps or 'plot_parcellation' in steps:
        print("Plotting Parcellation...\n")
        plot_parcellation(
            output_dir=cfg.get('output_dir', OUTPUT_DIR)
        )

    if 'all' in steps or 'plot_stability' in steps:
        print("Plotting Stability...\n")
        plot_stability(
            output_dir=cfg.get('output_dir', OUTPUT_DIR)
        )

    if 'all' in steps or 'subnetwork_knockout' in steps:
        print("Running Knockout...\n")
        run_knockout(
            output_dir=cfg.get('output_dir', OUTPUT_DIR),
            connectivity_kwargs=cfg.get('connectivity', {}),
            **cfg.get('subnetwork_extraction', {})
        )
    
    if 'all' in steps or 'run_sequential_knockout' in steps:
        print("Running Sequential Knockout...\n")
        run_sequential_knockout(
            output_dir=cfg.get('output_dir', OUTPUT_DIR),
            connectivity_kwargs=cfg.get('connectivity', {}),
            **cfg.get('run_sequential_knockout', {})
    )

    if 'all' in steps or 'random_knockout' in steps:
        print("Running Random Knockout...\n")
        run_random_perturbation(
            output_dir=cfg.get('output_dir', OUTPUT_DIR),
            connectivity_kwargs=cfg.get('connectivity', {}),
            **cfg.get('random_knockout', {})
    )


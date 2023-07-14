import logging
import os
import pathlib

import torch

from src.models.Baselines import test_baselines
from src.util.ParameterHandling import parse_args
from src.util.misc import set_seed

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
LOGGER = logging.getLogger()


def main():
    LOGGER.info('Starting main')
    args = parse_args()

    # paths
    base_dir = pathlib.Path(__file__).parent
    # results_dir
    results_dir = os.path.join(base_dir, 'results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    # plot_dir
    plot_dir = os.path.join(results_dir, 'plots')
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    # data_dir
    if args.data_dir == '':
        base_data_dir = os.path.join(base_dir, 'data')
    else:
        base_data_dir = args.data_dir
    # models_dir
    models_dir = os.path.join(base_dir, 'runs')
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    LOGGER.debug(f'Using device {device}')

    set_seed(args.seed)

    test_baselines(args, base_data_dir, models_dir, results_dir, args.img_size, device, batch_size=args.batch_size,
                   disable_tqdm=False)


if __name__ == '__main__':
    main()

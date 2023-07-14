import argparse
import logging
import os
import pathlib

import torch

from src.util.SelfSupervisedZeroShot import test_self_supervised_zero_shot

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
LOGGER = logging.getLogger()


def get_args_parser():
    parser = argparse.ArgumentParser('Self-supervsed semantic counting', add_help=False)

    # Model parameters
    parser.add_argument('--model_id', default=0, type=int, help='Id of the model to be loaded. If not specified, a trained CounTR model will be loaded.')
    parser.add_argument('--model_dir', default='', type=str, help='Path to a saved model. If not specified, a trained CounTR model will be loaded.')

    # Dataset parameters
    parser.add_argument('--img_dir', default='', type=str, help='path to the directory with the images')

    parser.add_argument('--notebook', action='store_true', help='Create visualisations for the jupyter notebook.')

    return parser


def main():

    LOGGER.info('Starting main')
    args = get_args_parser()
    args = args.parse_args()

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
    base_data_dir = os.path.join(base_dir, 'data')

    # models_dir
    models_dir = os.path.join(base_dir, 'runs')
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    LOGGER.debug(f'Using device {device}')

    test_self_supervised_zero_shot(args.model_id, models_dir, device, base_data_dir, args.img_dir, model_dir=args.model_dir,
                                   notebook=args.notebook)


if __name__ == '__main__':
    main()

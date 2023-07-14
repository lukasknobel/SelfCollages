import json
import logging
import os
import pathlib

import torch
from torch.utils.tensorboard import SummaryWriter

from src.data_handling.datasets.SelfCollageDataset import SelfCollageDataset
from src.models.ModelFactory import create_model
from src.train_evaluation_handling.TrainHandling import train_model
from src.util import Constants
from src.util.ParameterHandling import parse_args, get_model_dict, get_model_dir
from src.util.losses import DensityMSELoss
from src.util.misc import set_seed

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
LOGGER = logging.getLogger()

MODEL_DICT_FILE = 'hparams.json'


def main():
    LOGGER.info('Starting main')
    args = parse_args()
    model_dict = get_model_dict(args)

    LOGGER.info(f'Configuration: {json.dumps(model_dict, indent=4)}')

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

    # ImgNet directory
    if args.img_net_path == '':
        args.img_net_path = os.path.join(base_data_dir, 'ImageNet')

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    LOGGER.debug(f'Using device {device}')

    set_seed(args.seed)

    model_dir = get_model_dir(args, models_dir, model_dict, MODEL_DICT_FILE)

    model = create_model(args, weights_dir=base_data_dir)
    criterion = DensityMSELoss(args.density_scaling, args.density_loss_mask_prob,
                               args.density_loss_use_independent_masks, args.density_loss_keep_object_pixels,
                               args.density_loss_keep_all_object_pixels,
                               args.density_loss_penalise_wrong_cluster_objects,
                               args.density_loss_wrong_cluster_penality, args.img_size)

    LOGGER.info(f'Self-supervised training of {model}')

    if args.normalise:
        normalise_transform = model.backbone.normalise_transform
    else:
        normalise_transform = None
    model = model.to(device)
    model_path = os.path.join(model_dir, f'{model}.pt')
    args_path = os.path.join(model_dir, Constants.model_args_file_name)
    summary_writer = SummaryWriter(os.path.join(model_dir, Constants.TB_SUB_DIR))

    # return saved model if it was already trained
    if os.path.isfile(model_path):
        LOGGER.info(f'Returning saved trained {model}')
        with open(model_path, 'rb') as f:
            state_dict = torch.load(f, map_location=device)
        model.load_state_dict(state_dict)
    else:
        with open(args_path, 'wb') as f:
            LOGGER.info(f'Saving arguments')
            torch.save(args, f)
        # training
        patch_size = model.patch_size
        if patch_size is None:
            patch_size = 16
        training_dataset = SelfCollageDataset(args, base_data_dir, normalise_transform=normalise_transform,
                                              device=device, patch_size=patch_size, plot_dir=plot_dir,
                                              weights_dir=base_data_dir)
        # measure indexing time of training dataset
        training_dataset.measure_dataset_indexing_time(disable_tqdm=args.disable_tqdm)

        model = train_model(args, model_path, model, training_dataset, criterion=criterion,
                            num_count_classes=args.num_count_classes, summary_writer=summary_writer, device=device,
                            eval_func=None, visualise_test_samples=None)

        with open(model_path, 'wb') as f:
            LOGGER.info(f'Saving trained {model}')
            torch.save(model.state_dict(), f)
    LOGGER.info(f'Model directory: {model_dir}')


if __name__ == '__main__':
    main()

import os

import torch

from . import Constants
from .ParameterHandling import fill_missing_values_with_defaults
from ..models.ModelFactory import create_model


def load_model(model_dir, device, weights_dir, num_obj_classes=None):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f'Model directory {model_dir} does not exist')

    model_args_path = os.path.join(model_dir, Constants.model_args_file_name)
    if not os.path.isfile(model_args_path):
        raise FileNotFoundError(f'Model arguments file {model_args_path} does not exist')
    model_args = torch.load(model_args_path)

    filled_args = fill_missing_values_with_defaults(model_args)

    model = create_model(filled_args, num_obj_classes=num_obj_classes, weights_dir=weights_dir)
    if isinstance(model, tuple):
        model,  = model

    model = model.to(device)
    model_path = os.path.join(model_dir, f'{model}.pt')

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Model file {model_path} does not exist')
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=device)
    model.load_state_dict(state_dict)

    return model, model_args

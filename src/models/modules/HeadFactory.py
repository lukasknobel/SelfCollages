import logging

import torch
from torch import nn

from . import MLP
from .DensityMapDecoder import DensityMapDecoder
from .SplitDensityMapDecoder import SplitDensityMapDecoder
from ..ModelEnums import Heads

LOGGER = logging.getLogger()


def _get_regression_act_fn(regression_act_fn_name):
    if regression_act_fn_name.lower() == 'relu':
        regression_act_fn = nn.ReLU()
    elif regression_act_fn_name.lower() == 'square':
        class Square(nn.Module):
            def forward(self, x):
                return torch.square(x)
        regression_act_fn = Square()
    elif regression_act_fn_name.lower() == 'identity':
        regression_act_fn = nn.Identity()
    else:
        raise ValueError(f'Activation function {regression_act_fn_name} is not supported')
    return regression_act_fn


def _get_scalar_head(args, in_features, num_mlp_layers, mlp_hidden_dim):
    regression_act_fn = _get_regression_act_fn(args.regression_act_fn)

    if args.head is Heads.LINEAR:
        return nn.Sequential(nn.Linear(in_features, 1), regression_act_fn)
    elif args.head is Heads.MLP:
        return nn.Sequential(MLP(in_features, mlp_hidden_dim, 1, num_mlp_layers), regression_act_fn)
    else:
        raise ValueError(f'Head {args.head.name} is not supported')


def get_head(args, in_features, upsampling_factor, num_mlp_layers=3, mlp_hidden_dim=256):

    if args.split_map_and_count:
        map_decoder = DensityMapDecoder(in_features, upsampling_factor)
        count_head = _get_scalar_head(args, in_features, num_mlp_layers, mlp_hidden_dim)
        return SplitDensityMapDecoder(map_decoder, count_head, args.density_scaling)

    LOGGER.info(f'Ignoring specified head ({args.head.name}) when using density map')
    return DensityMapDecoder(in_features, upsampling_factor)

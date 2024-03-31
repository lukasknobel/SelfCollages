import logging

from .ModelEnums import ModelTypes
from .UnCounTRModel import UnCounTRModel

LOGGER = logging.getLogger()


def create_model(args, **kwargs):

    # create the specified model
    if args.model_type is ModelTypes.UnCounTR:
        model = UnCounTRModel(args, **kwargs)
    else:
        raise ValueError(f'model type {args.model_type.name} unknown')

    return model

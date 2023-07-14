import logging

from .ModelEnums import ModelTypes
from .UnCoModel import UnCoModel

LOGGER = logging.getLogger()


def create_model(args, **kwargs):

    # create the specified model
    if args.model_type is ModelTypes.UnCo:
        model = UnCoModel(args, **kwargs)
    else:
        raise ValueError(f'model type {args.model_type.name} unknown')

    return model

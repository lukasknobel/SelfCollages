from enum import Enum


class SupportedEvalDatasets(Enum):
    # dataset combinations
    BaselineDefault = 1
    EvalDefault = 8
    # individual datasets
    FSC147 = 2
    FSC147_val = 3
    FSC147_low = 4
    FSC147_medium = 5
    FSC147_high = 6
    MSO_few_shot = 7

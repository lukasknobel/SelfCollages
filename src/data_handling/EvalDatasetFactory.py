from .DatasetSplits import DatasetSplits
from .SupportedDatasets import SupportedEvalDatasets
from .datasets.FSCDataset import FSCDataset
from .datasets.MSODataset import MSODataset
from .datasets.CARPKDataset import CARPKDataset


def create_eval_dataset(eval_dataset, base_data_dir, **kwargs):
    # create the specified model
    if eval_dataset is SupportedEvalDatasets.MSO_few_shot:
        eval_dataset = MSODataset(base_data_dir, split=DatasetSplits.TEST, few_shot=True, **kwargs)
    elif eval_dataset is SupportedEvalDatasets.FSC147:
        eval_dataset = FSCDataset(base_data_dir, use_133_subset=False, **kwargs)
    elif eval_dataset is SupportedEvalDatasets.FSC147_val:
        eval_dataset = FSCDataset(base_data_dir, use_133_subset=False, split=DatasetSplits.VAL, **kwargs)
    elif eval_dataset is SupportedEvalDatasets.FSC147_low:
        eval_dataset = FSCDataset(base_data_dir, use_133_subset=False, max_count=17, **kwargs)
    elif eval_dataset is SupportedEvalDatasets.FSC147_medium:
        eval_dataset = FSCDataset(base_data_dir, use_133_subset=False, min_count=17, max_count=41, **kwargs)
    elif eval_dataset is SupportedEvalDatasets.FSC147_high:
        eval_dataset = FSCDataset(base_data_dir, use_133_subset=False, min_count=41, **kwargs)
    elif eval_dataset is SupportedEvalDatasets.CARPK:
        eval_dataset = CARPKDataset(base_data_dir, **kwargs)
    else:
        raise ValueError(f'Evaluation dataset {eval_dataset.name} is not supported')

    return eval_dataset

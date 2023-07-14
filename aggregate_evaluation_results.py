import argparse
import datetime
import logging
import os
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from src.data_handling.EvalDatasetFactory import create_eval_dataset
from src.data_handling.SupportedDatasets import SupportedEvalDatasets

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
LOGGER = logging.getLogger()


def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation Aggregator', add_help=False)

    parser.add_argument('--eval_results_path', default='', type=str, help='path to the eval results')
    parser.add_argument('--data_dir', default='', type=str, help='path to the base directory containing the datasets')
    parser.add_argument('--dataset_type', default='test', type=str, help='test, val or mso')

    return parser


def main():

    args = get_args_parser()
    args = args.parse_args()

    # paths
    base_dir = Path(__file__).parent
    # results_dir
    results_dir = os.path.join(base_dir, 'results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    # data_dir
    if args.data_dir == '':
        base_data_dir = os.path.join(base_dir, 'data')
    else:
        base_data_dir = args.data_dir

    dir_elements = [Path(args.eval_results_path, el) for el in os.listdir(args.eval_results_path)]

    if len(dir_elements) == 0:
        LOGGER.info(f'No files found in {args.eval_results_path}')
        return
    if dir_elements[0].is_dir():
        LOGGER.info(f'First element in {args.eval_results_path} is a directory, assuming all results are in subdirectories')
        experiments_paths = dir_elements
    else:
        experiments_paths = [Path(args.eval_results_path)]

    if args.dataset_type == 'test':
        fsc_subsets = [SupportedEvalDatasets.FSC147, SupportedEvalDatasets.FSC147_low, SupportedEvalDatasets.FSC147_medium,
                       SupportedEvalDatasets.FSC147_high]
    elif args.dataset_type == 'val':
        fsc_subsets = [SupportedEvalDatasets.FSC147_val]
    elif args.dataset_type == 'mso':
        fsc_subsets = [SupportedEvalDatasets.MSO_few_shot]

    fsc_datasets = {}
    for fsc_subset in fsc_subsets:
        fsc_datasets[fsc_subset.name] = create_eval_dataset(fsc_subset, base_data_dir, in_memory=False)

    metrics = ['mae', 'rmse', 'kendalltau']
    header = ['run'] + [f'{fsc_subset.name}_{metric}' for fsc_subset in fsc_subsets for metric in metrics]
    all_results = []
    for experiment_path in experiments_paths:
        results_path = os.path.join(experiment_path, 'results.csv')
        if not os.path.isfile(results_path):
            LOGGER.info(f'No results.csv found in {experiment_path}')
            continue
        str_results = np.loadtxt(os.path.join(experiment_path, 'results.csv'), str)
        # pred_ids = str_results[0].astype(int)
        if args.dataset_type == 'mso':
            pred_img_names = [int(s.split('_')[1].split('.')[0]) for i, s in enumerate(str_results[1])]
        else:
            pred_img_names = {str(s): i for i, s in enumerate(str_results[1])}
        preds = str_results[2].astype(float)
        run_results = [experiment_path.name]
        for subset_name, subset in fsc_datasets.items():
            subset_files = subset.file_names
            subset_labels = subset.labels
            if args.dataset_type == 'mso':
                subset_labels = [subset_labels[pred_img_names[i]] for i in range(len(pred_img_names))]
                subset_preds = preds
            else:
                subset_preds = [preds[pred_img_names[p_img]] for p_img in subset_files]

            subset_mae = np.mean(np.abs(np.array(subset_preds) - np.array(subset_labels)))
            subset_rmse = np.sqrt(np.mean(np.square(np.array(subset_preds) - np.array(subset_labels))))
            subset_kendalltau = kendalltau(np.array(subset_labels), np.array(subset_preds)).correlation
            run_results += [subset_mae, subset_rmse, subset_kendalltau]
        all_results.append(run_results)
    with open(os.path.join(results_dir, f'{Path(args.eval_results_path).name }_{len(all_results)}_{args.dataset_type}_results_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.csv'), 'w') as f:
        np.savetxt(f, all_results, header=','.join(header), delimiter=',', fmt='%s')


if __name__ == '__main__':
    main()

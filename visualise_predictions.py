import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from IPython.core.display_functions import display
from matplotlib import pyplot as plt, ticker

from src.data_handling.EvalDatasetFactory import create_eval_dataset
from src.data_handling.SupportedDatasets import SupportedEvalDatasets

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
LOGGER = logging.getLogger()


def get_args_parser():
    parser = argparse.ArgumentParser('Visualise predictions', add_help=False)

    parser.add_argument('--eval_results_path', default='', type=str, help='path to the eval results')
    parser.add_argument('--data_dir', default='', type=str, help='path to the base directory containing the datasets')
    parser.add_argument('--dataset_type', default='test', type=str, help='test, val or mso')

    parser.add_argument('--notebook', action='store_true', help='Create visualisations for the jupyter notebook.')

    return parser


def visualise_preds(args, experiment_path, base_data_dir, results_dir, max_plot_label=None, use_log_scale=True,
                    plot_average_pred=False, plot_trend_line=False, scatter_alpha=0.7):

    if args.dataset_type == 'test':
        fsc_subsets = [SupportedEvalDatasets.FSC147_low,
                       SupportedEvalDatasets.FSC147_medium,
                       SupportedEvalDatasets.FSC147_high]
    elif args.dataset_type == 'val':
        fsc_subsets = [SupportedEvalDatasets.FSC147_val]
    elif args.dataset_type == 'mso':
        fsc_subsets = [SupportedEvalDatasets.MSO_few_shot]
    else:
        raise ValueError(f'Unknown dataset type {args.dataset_type}')

    fsc_datasets = {}
    for fsc_subset in fsc_subsets:
        fsc_datasets[fsc_subset.name] = create_eval_dataset(fsc_subset, base_data_dir, in_memory=False)

    str_results = np.loadtxt(os.path.join(experiment_path, 'results.csv'), str)
    if args.dataset_type == 'mso':
        pred_img_names = [int(s.split('_')[1].split('.')[0]) for i, s in enumerate(str_results[1])]
    else:
        pred_img_names = {str(s): i for i, s in enumerate(str_results[1])}
    preds = str_results[2].astype(float)

    plt.style.use("seaborn-v0_8")
    font_size = 18
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('legend', fontsize=font_size)
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    all_labels = []
    all_preds = []
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax1.set_facecolor('w')
    ax2.set_facecolor('w')
    ax1.yaxis.grid(color='lightgray')
    ax1.xaxis.grid(color='lightgray')
    ax2.yaxis.grid(color='lightgray')
    ax2.xaxis.grid(color='lightgray')

    for i, (subset_name, subset) in enumerate(fsc_datasets.items()):
        subset_files = subset.file_names
        subset_labels = subset.labels
        if args.dataset_type == 'mso':
            subset_labels = torch.tensor([subset_labels[pred_img_names[i]] for i in range(len(pred_img_names))])
            subset_preds = preds
        else:
            subset_preds = np.array([preds[pred_img_names[p_img]] for p_img in subset_files])
        if max_plot_label is not None:
            mask = subset_labels <= max_plot_label
            subset_labels = subset_labels[mask]
            subset_preds = subset_preds[:, mask]
        all_labels.append(subset_labels)
        all_preds.append(subset_preds)
        if args.dataset_type == 'test':
            subset_name = subset_name.replace('_', ' ')
            subset_name = f'{subset_name[:3]}-{subset_name[3:]}'
        else:
            subset_name = None
        ax1.scatter(subset_labels, subset_preds, marker='.', c=colours[i], alpha=scatter_alpha, label=subset_name)
        ax2.scatter(subset_labels, (torch.tensor(subset_preds) - subset_labels) / subset_labels, marker='.',
                    c=colours[i], alpha=scatter_alpha, label=subset_name)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    min_label = min(all_labels)
    max_label = max(all_labels)
    next_colour = plt.rcParams['text.color']
    ax1.plot([0, max_label], [0, max_label], '-', color=next_colour, label='ground-truth')
    # add trendline to plot
    if plot_trend_line:
        z = np.polyfit(all_labels, all_preds, 1)
        p = np.poly1d(z)
        ax1.plot([min_label, max_label], [p(min_label), p(max_label)], '-', color='k', label='trend line')
    # add average prediction
    if plot_average_pred:
        avg_pred = np.array(all_preds).mean()
        ax1.plot([0, max_label], [avg_pred, avg_pred], '--', color=colours[i + 1], label='average prediction')
    avg_relative_err = ((torch.tensor(all_preds) - all_labels) / all_labels).mean()
    ax2.plot([0, max_label], [avg_relative_err, avg_relative_err], '-', color=next_colour,
             label='average error')
    if use_log_scale:
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')
    ax1.set_xlabel('annotated count')
    ax1.set_ylabel('prediction')
    ax1.legend(markerscale=2.5)
    fig1.tight_layout()
    fig1.savefig(os.path.join(results_dir, 'predictions.pdf'), bbox_inches='tight')
    if args.notebook:
        display(fig1)
    else:
        fig1.show()
    ax2.set_xlabel('annotated count')
    ax2.set_ylabel('relative error')
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax2.legend(markerscale=2.5)
    fig2.tight_layout()
    fig2.savefig(os.path.join(results_dir,'relative_error.pdf'), bbox_inches='tight')
    if args.notebook:
        display(fig2)
    else:
        fig2.show()
    plt.style.use("default")


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
        LOGGER.info(
            f'First element in {args.eval_results_path} is a directory, assuming all results are in subdirectories. Using first subdirectory.')
        experiment_path = dir_elements[0]
    else:
        experiment_path = Path(args.eval_results_path)

    LOGGER.debug(f'Visualising predictions for {experiment_path}')

    visualise_preds(args, experiment_path, base_data_dir, results_dir, max_plot_label=None, use_log_scale=True,
                    plot_average_pred=False, plot_trend_line=False)


if __name__ == '__main__':
    main()

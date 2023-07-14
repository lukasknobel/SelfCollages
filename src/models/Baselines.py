import datetime
import json
import os
from pathlib import Path

import matplotlib.patheffects as PathEffects
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import label
from scipy.stats import kendalltau
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from tqdm import tqdm

from .ModelEnums import BackboneTypes, PretrainedWeights
from .backbones import Backbone, BackboneFreezing
from ..data_handling.DatasetSplits import DatasetSplits
from ..data_handling.EvalDatasetFactory import create_eval_dataset
from ..data_handling.SupportedDatasets import SupportedEvalDatasets
from ..data_handling.datasets.CountingDataset import CountingDataset
from ..data_handling.datasets.FSCDataset import FSCDataset
from ..train_evaluation_handling.EvaluationHandling import evaluate_models_on_datasets, load_evaluation_results, \
    save_evaluation_results
from ..util import transforms, plotting
from ..util.misc import get_resize_and_cropping_transforms


class AverageBaseline(nn.Module):
    def __init__(self, static_pred=None):
        super().__init__()
        self.static_pred = static_pred

    def forward(self, x, *args):
        if self.static_pred is None:
            raise ValueError(f'Specify pred_class before using {self.__class__.__name__} for prediction')
        preds = torch.ones((x.shape[0],), device=x.device) * self.static_pred

        return preds

    def __repr__(self):
        if self.static_pred is None:
            return f'AverageBaseline'
        else:
            return f'AverageBaseline_{self.static_pred}'


class DINOBaseline(nn.Module):
    def __init__(self, weights_dir, target_img_size):
        super().__init__()

        self.target_img_size = target_img_size

        self.att_model = Backbone(BackboneTypes.ViT_B_8, PretrainedWeights.DINO, weights_dir,
                                  return_feature_vector=False,
                                  return_cls_self_attention=False,
                                  backbone_freezing=BackboneFreezing.completely)

        # FSC regression
        # parameters based on the FSC training set
        if self.target_img_size == 384:
            self.att_threshold, self.att_head_threshold, self.min_size_percentage = (0.7, 10, 0)
        else:
            # parameters based on 224x224
            self.att_threshold, self.att_head_threshold, self.min_size_percentage = (0.2, 0, 0)

        self.w_featmap = self.target_img_size // self.att_model.patch_size

    @property
    def min_obj_size(self):
        return int((self.target_img_size//self.att_model.patch_size)**2 * self.min_size_percentage)

    def find_att_threshold(self, base_data_dir, device, num_images=None, uniform_distr=True, random_samples=False,
                           num_workers=0, batch_size=32, disable_tqdm=False):
        trans, _ = get_resize_and_cropping_transforms(self.target_img_size, cropping=False)
        training_transforms = transforms.Compose(trans)
        training_dataset = FSCDataset(base_data_dir, split=DatasetSplits.TRAIN, transform=training_transforms, in_memory=True)
        classes, num_imgs_per_class = training_dataset.labels.unique(return_counts=True)
        if num_images is None:
            selected_idxs = torch.arange(len(training_dataset))
        else:
            if uniform_distr and random_samples:
                raise ValueError('uniform_distr and random_samples cannot both be True')
            if random_samples:
                selected_idxs = torch.randperm(len(training_dataset))[:num_images]
            else:
                if uniform_distr:
                    subset_imgs_per_class = torch.ones(classes.shape[0], dtype=torch.int) * (num_images // classes.shape[0])
                    num_missing = num_images - subset_imgs_per_class.sum().int()
                    subset_imgs_per_class[classes.shape[0]-num_missing:] += 1
                else:
                    label_distr = num_imgs_per_class/num_imgs_per_class.sum()
                    subset_imgs_per_class = (label_distr*num_images).int()
                    subset_imgs_per_class[subset_imgs_per_class.argmax()] += num_images - subset_imgs_per_class.sum()
                selected_idxs = []
                for cls, subset_imgs in zip(classes, subset_imgs_per_class):
                    possible_cls_idxs = torch.arange(len(training_dataset))[training_dataset.labels == cls]
                    selected_cls_idxs = possible_cls_idxs[torch.randperm(possible_cls_idxs.shape[0])[:subset_imgs]]
                    selected_idxs.append(selected_cls_idxs)
                selected_idxs = torch.concatenate(selected_idxs)
        training_subset_labels = training_dataset.labels[selected_idxs].to(device)
        training_subset_dataset = Subset(training_dataset, selected_idxs)
        self.att_model.to(device)
        results_dict = {(att_threshold, att_head_threshold, min_size_percentage): None for min_size_percentage in [0, 0.01, 0.02, 0.05, 0.1, 0.2] for att_threshold in [0.1 + i * 0.1 for i in range(9)]+[0.95, 0.99] for att_head_threshold in [i for i in range(13)]}
        best_config = None
        best_result = None
        with torch.no_grad():
            training_dataloader = DataLoader(training_subset_dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True)
            nh = 0
            total_idx = []
            total_cumval = []
            total_att = []
            for (imgs, _) in tqdm(training_dataloader, disable=disable_tqdm):
                imgs = imgs.to(device)
                nh, idx, cumval, att = self._get_att_map(imgs)
                total_idx.append(idx.cpu())
                total_cumval.append(cumval.cpu())
                total_att.append(att.cpu())
            total_idx = torch.concat(total_idx)
            total_cumval = torch.concat(total_cumval)
            total_att = torch.concat(total_att)

            for config in tqdm(results_dict.keys(), disable=disable_tqdm):
                att_threshold, att_head_threshold, min_size_percentage = config
                self.att_threshold = att_threshold
                self.att_head_threshold = att_head_threshold
                self.min_size_percentage = min_size_percentage
                preds = self._get_pred(training_subset_labels.shape[0], nh, total_idx, total_cumval, total_att, device)

                # MAE
                cur_result = (training_subset_labels - preds).abs().mean().cpu()
                results_dict[config] = f'{round(cur_result.item(), 4)}'

                new_best = best_result is None
                if not new_best:
                    new_best = cur_result < best_result
                if new_best:
                    best_result = cur_result
                    best_config = config

        print(best_config, results_dict[best_config])
        print(json.dumps({str(k): v for k, v in results_dict.items()}, indent=4))

    def _get_att_map(self, x):
        if self.att_model.normalise_transform is not None:
            x = self.att_model.normalise_transform(x)

        # determine object box based on self attention map
        # based on https://github.com/facebookresearch/dino/blob/main/visualize_attention.py
        att = self.att_model.model.get_last_selfattention(x)
        nh = att.shape[1]
        t = att[:, :, 0, 1:].reshape(x.shape[0], nh, -1)

        # only keep a certain percentage of the mass
        val, idx = torch.sort(t)
        val /= torch.sum(val, dim=-1, keepdim=True)
        cumval = torch.cumsum(val, dim=-1)
        return nh, idx, cumval, t

    def _get_pred(self, num_imgs, nh, idx, cumval, att, device):
        preds = torch.zeros((num_imgs,), device=device)

        th_attn = cumval > 1 - self.att_threshold
        idx2 = torch.argsort(idx)

        img_idxs = torch.arange(num_imgs).repeat_interleave(nh)
        head_idxs = torch.arange(nh).repeat(num_imgs)
        th_attn[img_idxs, head_idxs] = th_attn[img_idxs, head_idxs][
            torch.arange(img_idxs.shape[0]).view(-1, 1), idx2[img_idxs, head_idxs]]

        th_attn = th_attn.reshape(num_imgs, nh, self.w_featmap, self.w_featmap).float()
        th_attn_agg_heads = (th_attn.sum(1) > self.att_head_threshold).cpu()

        att = att.sum(1)
        att_val, att_idx = torch.sort(att)
        att_val /= torch.sum(att_val, dim=-1, keepdim=True)
        att_cumval = torch.cumsum(att_val, dim=-1)
        th_sum_attn = att_cumval > 1 - self.att_threshold
        att_idx2 = torch.argsort(att_idx)
        th_sum_attn[img_idxs] = th_sum_attn[img_idxs][
            torch.arange(img_idxs.shape[0]).view(-1, 1), att_idx2[img_idxs]]

        for i in range(num_imgs):

            # get number of connected components
            connected_components, num_components = label(th_attn_agg_heads[i], structure=np.ones((3, 3)))
            component_sizes = torch.tensor(connected_components).unique(return_counts=True)[1][1:]
            num_components = (component_sizes > self.min_obj_size).sum().item()
            preds[i] = num_components
        return preds

    def forward(self, x, *args):
        nh, idx, cumval, att = self._get_att_map(x)

        preds = self._get_pred(x.shape[0], nh, idx, cumval, att, x.device)
        return preds

    def __repr__(self):
        return f'{self.__class__.__name__}_{self.target_img_size}'


class FasterRCNNBaseline(nn.Module):
    def __init__(self, target_img_size, box_score_thresh=0.5):
        super().__init__()
        self.model_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.box_score_thresh = box_score_thresh
        self.target_img_size = target_img_size
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.model_weights, box_score_thresh=self.box_score_thresh)
        self.model.eval()

    def forward(self, x, *args, return_boxes=False):

        x = self.model_weights.transforms()(x)

        pred_dicts = self.model(x)

        pred_boxes = [pred_dict['boxes'] for pred_dict in pred_dicts]

        preds = torch.tensor([pred_box.shape[0] for pred_box in pred_boxes], device=x.device)

        if return_boxes:
            return preds, pred_boxes
        else:
            return preds

    def visualise_wrong_preds(self, base_data_dir, base_plot_dir, target_img_size, device, samples=None, batch_size=16,
                              max_num_plots=10, num_classes=5, num_workers=0, regression=True,
                              plot_labeled_boxes=False, save_figure=True, show_figure=True):

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(target_img_size, target_img_size))
        ])

        plot_dir = os.path.join(base_plot_dir, 'model_predictions', str(self))
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        if samples is None:
            test_dataset = create_eval_dataset(SupportedEvalDatasets.FSC147, base_data_dir, transform=test_transforms, in_memory=True)

            data_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True, shuffle=True)
            self.to(device)

            remaining_plots = max_num_plots
            with torch.no_grad():
                for imgs, targets in data_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
                    preds, pred_boxes = self.forward(imgs, return_boxes=True)
                    labels = targets[CountingDataset.LABEL_DICT_GLOBAL_COUNT]
                    wrong_idxs = torch.arange(preds.shape[0])[(preds != labels).cpu()]

                    if wrong_idxs.shape[0] > remaining_plots:
                        wrong_idxs = wrong_idxs[torch.randperm(wrong_idxs.shape[0])[:remaining_plots]]
                    remaining_plots -= wrong_idxs.shape[0]
                    for i, wrong_idx in enumerate(wrong_idxs):
                        fig = plt.figure(figsize=(6,6))
                        plt.title(f'Label: {labels[wrong_idx]}, prediction: {preds[wrong_idx]}')
                        plt.imshow(imgs[wrong_idx].permute(1,2,0).cpu())
                        plotting.plot_boxes(pred_boxes[wrong_idx].cpu(), fig)
                        if plot_labeled_boxes:
                            labeled_boxes = CountingDataset.unpad_sample(targets[CountingDataset.LABEL_DICT_BOXES][wrong_idx].cpu())
                            labeled_boxes[:, [0, 2]] *= imgs[i].shape[-1]
                            labeled_boxes[:, [1, 3]] *= imgs[i].shape[-2]
                            plotting.plot_boxes(labeled_boxes, fig, edgecolor='y', linestyles='--')

                        plt.axis('off')
                        plt.tight_layout()
                        if save_figure:
                            plt.savefig(os.path.join(plot_dir, f'wrong_pred_{remaining_plots+i}.jpg'))
                        if show_figure:
                            plt.show()
                    if remaining_plots <= 0:
                        break
        else:
            for name, orig_img, targets in samples:
                img, targets = test_transforms(orig_img, targets)
                orig_img = transforms.ToTensor()(orig_img)
                img = img.to(device, non_blocking=True)
                targets = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in
                           targets.items()}
                preds, pred_boxes = self.forward(img.unsqueeze(0), return_boxes=True)
                pred_boxes = pred_boxes[0].detach().cpu()
                pred_boxes[:, [0, 2]] *= orig_img.shape[-1] / img.shape[-1]
                pred_boxes[:, [1, 3]] *= orig_img.shape[-2] / img.shape[-2]

                plt.rcParams.update({'font.size': 40})
                h, w = orig_img.shape[1:]
                factor = 50
                fig = plt.figure(figsize=(w / factor, h / factor))
                plt.imshow(orig_img.permute(1, 2, 0).cpu())
                plotting.plot_boxes(pred_boxes, fig, linewidths=6)
                if plot_labeled_boxes:
                    labeled_boxes = CountingDataset.unpad_sample(targets[CountingDataset.LABEL_DICT_BOXES].cpu())
                    labeled_boxes[:, [0, 2]] *= orig_img.shape[-1]
                    labeled_boxes[:, [1, 3]] *= orig_img.shape[-2]
                    plotting.plot_boxes(labeled_boxes, fig, edgecolor='y', linestyles='--')
                str_pred_cnt = str(round(pred_boxes.shape[0]))
                txt = plt.text(10, 50, str_pred_cnt, c='w', fontsize='xx-large')
                txt.set_path_effects([PathEffects.withStroke(linewidth=6, foreground='k')])
                plt.axis('off')

                plt.tight_layout()
                if save_figure:
                    plt.savefig(os.path.join(plot_dir, f'wrong_pred_{name}'))
                if show_figure:
                    plt.show()
                plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})

    def __repr__(self):
        return f'{self.__class__.__name__}_{self.box_score_thresh}_{self.target_img_size}'


def get_subset_results(base_data_dir, results_dir, paths):

    fsc_subsets = [SupportedEvalDatasets.FSC147_low, SupportedEvalDatasets.FSC147_medium,
               SupportedEvalDatasets.FSC147_high]
    fsc_datasets = {}
    for fsc_subset in fsc_subsets:
        fsc_datasets[fsc_subset.name] = create_eval_dataset(fsc_subset, base_data_dir, in_memory=False)

    metrics = ['mae', 'rmse', 'r2', 'kendalltau']
    header = ['run'] + [f'{fsc_subset.name}_{metric}' for fsc_subset in fsc_subsets for metric in metrics]
    all_results = []
    for path in paths:
        pred_path = os.path.join(path, 'preds.pt')
        with open(pred_path, 'rb') as f:
            preds = torch.load(f)

        superset_preds, superset_targets = preds['FSC147']

        run_results = [Path(path).name]
        for subset_name, subset in fsc_datasets.items():
            mask = np.ones_like(superset_targets, dtype=bool)
            if subset.min_count is not None:
                mask &= superset_targets >= subset.min_count
            if subset.max_count is not None:
                mask &= superset_targets < subset.max_count
            subset_preds = superset_preds[mask]
            subset_labels = superset_targets[mask]

            subset_mae = np.mean(np.abs(np.array(subset_preds) - np.array(subset_labels)))
            subset_rmse = np.sqrt(np.mean(np.square(np.array(subset_preds) - np.array(subset_labels))))
            subset_r2 = r2_score(np.array(subset_labels), np.array(subset_preds))
            subset_kendalltau = kendalltau(np.array(subset_labels), np.array(subset_preds)).correlation
            run_results += [subset_mae, subset_rmse, subset_r2, subset_kendalltau]
        all_results.append(run_results)
    with open(os.path.join(results_dir, f'baseline_subset_results_for_{len(all_results)}_baselines_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.csv'), 'w') as f:
        np.savetxt(f, all_results, header=','.join(header), delimiter=',', fmt='%s')


def test_baselines(args, base_data_dir, models_dir, results_dir, target_img_size, device, batch_size=32, disable_tqdm=False):
    args.img_size = target_img_size

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(target_img_size, target_img_size))
    ])

    # use FSC train statistics for baselines, the two static baseline uses the mean of the whole and the small (<16 objects) dataset
    models = [AverageBaseline(static_pred=49.9631),
              DINOBaseline(base_data_dir, target_img_size), FasterRCNNBaseline(target_img_size)]
    seeds = [None for _ in range(len(models))]

    missing_model_dirs = []
    missing_models = []
    missing_seeds = []
    for model, seed in zip(models, seeds):
        model_dir = os.path.join(models_dir, str(model))
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        model_result = load_evaluation_results(model_dir)
        if model_result is None:
            missing_model_dirs.append(model_dir)
            missing_models.append(model)
            missing_seeds.append(seed)
    criteria = [None for _ in range(len(missing_models))]

    if len(missing_models) > 0:
        results, preds = evaluate_models_on_datasets(args, base_data_dir, SupportedEvalDatasets.BaselineDefault, missing_models, criteria, device,
                                              test_transforms=test_transforms, batch_size=batch_size, seeds=missing_seeds, disable_tqdm=disable_tqdm, return_last_preds=True)
        for path, pred in zip(missing_model_dirs, preds):
            with open(os.path.join(path, 'preds.pt'), 'wb') as f:
                torch.save(pred, f)

        save_evaluation_results(results, missing_model_dirs)

        get_subset_results(base_data_dir, results_dir, missing_model_dirs)

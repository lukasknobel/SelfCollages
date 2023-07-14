import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from . import transforms


def match_str_and_enum(str_, enum_, match_case=False):
    for enum_element in enum_:
        if match_case and enum_element.name == str_:
            return enum_element
        elif not match_case and enum_element.name.lower() == str_.lower():
            return enum_element
    raise ValueError(f'Unknown name \"{str_}\" for {enum_.__name__}. Supported values are: {", ".join([el.name for el in enum_])}')


def translate_enum_element_by_name(enum_element, enum_, match_case=False, default_enum_element=None):
    name = enum_element.name
    if not match_case:
        name = name.lower()
    for matched_enum_el in enum_:
        candidate_name = matched_enum_el.name
        if not match_case:
            candidate_name = candidate_name.lower()
        if name == candidate_name:
            return matched_enum_el
    if default_enum_element is None:
        raise ValueError(f'Could not find matching element for {enum_element} in {enum_}')
    else:
        return default_enum_element


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_dict_for_summary(hparams_dict):
    new_dict = {}
    for key, value in hparams_dict.items():
        if isinstance(value, dict):
            sub_dict = prepare_dict_for_summary(value)
            for sub_key, sub_value in sub_dict.items():
                new_dict[key+'_'+sub_key] = sub_value
        else:
            new_dict[key] = value
    return new_dict


def compute_intersection_and_union(box_a, box_b):
    box_pair = torch.stack([box_a, box_b])
    max_points = torch.max(box_pair, dim=0)[0]
    min_points = torch.min(box_pair, dim=0)[0]
    intersection = (min_points[2] - max_points[0]) * (min_points[3] - max_points[1])
    union = (max_points[2] - min_points[0]) * (max_points[3] - min_points[1])
    return intersection, union


@torch.no_grad()
def accuracy(preds, labels, top_k=(1,)):
    if isinstance(preds, dict):
        preds = preds['pred_logits']
    if len(preds.shape) == 1:
        preds = F.one_hot(preds)
    preds = preds.to(device=labels.device)
    _, top_k_classes = preds.topk(max(top_k), largest=True, sorted=True)
    correct_class = top_k_classes.eq(labels.view(-1, 1))
    accs = []
    for k in top_k:
        acc = correct_class[:, :k].any(dim=-1).float().mean()
        accs.append(acc)
    if len(accs) == 1:
        accs = accs[0]
    return accs

@torch.no_grad()
def mae_rmse(preds, labels):
    if isinstance(preds, dict):
        preds = preds['pred_logits']
    if len(preds.shape) == 2:
        pred_classes = torch.argmax(preds, dim=-1)
    else:
        pred_classes = preds
    mae = (pred_classes - labels).abs().float().mean()
    rmse = ((pred_classes-labels).float()**2).mean().sqrt()
    return mae, rmse

@torch.no_grad()
def construct_confusion_matrix(preds, labels, num_classes):
    if isinstance(preds, dict):
        preds = preds['pred_logits']
    if len(preds.shape) > 1:
        pred_classes = preds.argmax(dim=-1)
    else:
        pred_classes = preds
    pred_classes = pred_classes.to(device=labels.device)
    confusion_matrix = torch.zeros((num_classes, num_classes), device=preds.device)
    for c in range(num_classes):
        label_c = labels.eq(c)
        # confusion matrix
        for c2 in range(num_classes):
            confusion_matrix[c, c2] = sum(pred_classes[label_c].eq(c2))
    return confusion_matrix


def get_resize_and_cropping_transforms(target_img_size, cropping, cropping_ratio=0.875):
    if cropping:
        resize_img_size = int(target_img_size / cropping_ratio)
        crop_size = (target_img_size, target_img_size)
    else:
        resize_img_size = (target_img_size, target_img_size)

    trans = [transforms.ToTensor(), transforms.Resize(size=resize_img_size)]
    if cropping:
        trans += [transforms.RandomCrop(crop_size)]

    return trans, resize_img_size


def get_filtered_model_dirs(models_dir, model_names, include_baselines=True, ign_case=True, check_beginning=False,
                            return_sub_dirs=False):
    baseline_dirs = []
    baseline_sub_dirs = []
    model_dirs = defaultdict(list)
    sub_dirs = defaultdict(list)
    if ign_case:
        model_names = [model_name.lower() for model_name in model_names]
    for sub_dir in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, sub_dir)
        if not os.path.isdir(model_dir):
            continue
        if ign_case:
            loc_sub_dir = sub_dir.lower()
        else:
            loc_sub_dir = sub_dir
        if check_beginning:
            model_name_idx = [idx for idx, model_name in enumerate(model_names) if loc_sub_dir.startswith(model_name)]
        else:
            model_name_idx = any([idx for idx, model_name in enumerate(model_names) if model_name in loc_sub_dir])
        if len(model_name_idx) > 0:
            model_dirs[model_name_idx[0]].append(model_dir)
            sub_dirs[model_name_idx[0]].append(sub_dir)
        elif include_baselines:
            if 'baseline' in sub_dir.lower():
                baseline_dirs.append(model_dir)
                baseline_sub_dirs.append(sub_dir)

    final_model_dirs = baseline_dirs
    final_model_sub_dirs = baseline_sub_dirs
    for k in sorted(model_dirs.keys()):
        final_model_dirs.extend(model_dirs[k])
        final_model_sub_dirs.extend(sub_dirs[k])

    if return_sub_dirs:
        return final_model_dirs, final_model_sub_dirs
    else:
        return final_model_dirs



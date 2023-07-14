"""
Based on CounTR: https://github.com/Verg-Avesta/CounTR/blob/5bfacc7f837da10b3a8fc0e677264d62291900a2/FSC_finetune_cross.py#L166
"""
import logging

import numpy as np
import torch
from torch.nn.modules.loss import _Loss

from ...data_handling.datasets.CountingDataset import CountingDataset

LOGGER = logging.getLogger()


class DensityMSELoss(_Loss):
    def __init__(self, density_scaling, mask_prob=0.2, use_independent_masks=False, keep_object_pixels=False,
                 keep_all_object_pixels=False, penalise_wrong_cluster_objects=False, wrong_cluster_penality=1.0, standard_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.density_scaling = density_scaling
        self.mask_prob = (1-mask_prob)
        self.use_independent_masks = use_independent_masks
        self.keep_object_pixels = keep_object_pixels
        self.keep_all_object_pixels = keep_all_object_pixels
        self.penalise_wrong_cluster_objects = penalise_wrong_cluster_objects
        self.wrong_cluster_penality = wrong_cluster_penality
        self.logged_once = False
        self.standard_size = standard_size

    def forward(self, input_, target):
        if not isinstance(input_, torch.Tensor):
            raise NotImplementedError(f'Input must be a tensor not {type(input_)}.')
        if CountingDataset.LABEL_DICT_DENSITY_MAPS not in target:
            if not self.logged_once:
                self.logged_once = True
                LOGGER.warning(f'No density maps in target. Skipping loss calculation.')
            loss = None
        else:
            gt_density = target[CountingDataset.LABEL_DICT_DENSITY_MAPS]
            gt_all_density = target[CountingDataset.LABEL_DICT_ALL_DENSITY_MAPS]

            if self.use_independent_masks:
                # get a mask for each image in the batch
                masks = np.random.binomial(n=1, p=self.mask_prob, size=input_.shape)
                masks = torch.from_numpy(masks).to(input_.device)
            else:
                # get a single mask for the whole batch
                mask = np.random.binomial(n=1, p=self.mask_prob, size=[input_.shape[1], input_.shape[2]])
                masks = np.tile(mask, (input_.shape[0], 1))
                masks = masks.reshape(input_.shape)
                masks = torch.from_numpy(masks).to(input_.device)

            if self.keep_all_object_pixels:
                # always keep the pixels where there is any object
                has_any_obj = gt_all_density > 0
                masks = (masks.bool() | has_any_obj)
            elif self.keep_object_pixels:
                # always keep the pixels where there is a target object
                has_obj = gt_density > 0
                masks = (masks.bool() | has_obj)

            weighted_mask = masks.float()

            if self.penalise_wrong_cluster_objects:
                # apply a higher weight to pixels where there is a non-target object
                # this is just an approximation, which also increases the weight of some pixels where there is a target object
                wrong_obj_density = (gt_all_density - gt_density)/self.density_scaling > 1e-3
                weighted_mask[wrong_obj_density] = weighted_mask[wrong_obj_density] * self.wrong_cluster_penality

            loss = (input_ - gt_density) ** 2
            if self.standard_size is not None:
                # average over the standard size to make the loss independent of changing resolutions
                loss = ((loss * weighted_mask).sum(dim=(-2, -1)) / self.standard_size**2).mean()
            else:
                loss = (loss * weighted_mask).mean()

        pred_cnt = torch.sum(input_, dim=(1, 2)) / self.density_scaling
        pred_cnt = pred_cnt.clip(0)

        loss_dict = {'loss': loss, 'global_pred': pred_cnt, 'count_class_pred': pred_cnt.round().long(),
                     'count_scalar_pred': pred_cnt}

        return loss_dict

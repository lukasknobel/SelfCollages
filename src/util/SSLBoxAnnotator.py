import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import transforms
from .box_util import get_box_from_binary_mask
from .misc import get_resize_and_cropping_transforms
from .misc_enums import AnnotationTypes
from ..models.ModelEnums import PretrainedWeights, BackboneTypes
from ..models.backbones import BackboneFreezing, Backbone

segment_foreground = None

LOGGER = logging.getLogger()


class SSLBoxAnnotator:

    OBJ_BOX_FILE_PREFIX = 'obj_boxes'

    def __init__(self, annotation_type: AnnotationTypes, weights_dir, target_img_size, dino_att_threshold,
                 att_head_threshold, device, batch_size, num_workers, disable_tqdm):
        self.ann_type = annotation_type
        self.target_img_size = target_img_size
        self.dino_att_threshold = dino_att_threshold
        self.att_head_threshold = att_head_threshold
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.disable_tqdm = disable_tqdm

        if self.ann_type is AnnotationTypes.NO:
            self.ann_model = None
        elif self.ann_type is AnnotationTypes.DINO_SEG_8 or self.ann_type is AnnotationTypes.DINO_SEG_16:
            if self.ann_type is AnnotationTypes.DINO_SEG_16:
                self.ann_model = Backbone(BackboneTypes.ViT_B_16, PretrainedWeights.DINO, weights_dir,
                                              return_feature_vector=False,
                                              return_cls_self_attention=False,
                                              backbone_freezing=BackboneFreezing.completely)
            elif self.ann_type is AnnotationTypes.DINO_SEG_8:
                self.ann_model = Backbone(BackboneTypes.ViT_B_8, PretrainedWeights.DINO, weights_dir,
                                              return_feature_vector=False,
                                              return_cls_self_attention=False,
                                              backbone_freezing=BackboneFreezing.completely)
            if self.ann_model is not None:
                self.ann_model.to(self.device)
                self.ann_model.eval()
                self.att_head_threshold = max(
                    int(self.att_head_threshold * self.ann_model.model.blocks[-1].attn.num_heads), 1)
                LOGGER.debug(f'Attention head threshold is set to {self.att_head_threshold}')
        elif self.ann_type is AnnotationTypes.SPECTRAL_DINO_SEG_8 or self.ann_type is AnnotationTypes.SPECTRAL_DINO_SEG_16:
            self.ann_model = None
        else:
            raise ValueError(f'Unknown annotation type {self.ann_type.name}')

    @torch.no_grad()
    def annotate_boxes(self, dataset, dataset_ann_path):
        if self.ann_type is AnnotationTypes.NO:
            LOGGER.debug(
                f'Annotation type {self.ann_type.name} selected for dataset {dataset.__class__.__name__}. No annotation will be performed.')
            return None

        LOGGER.debug(
            f'Annotating boxes for dataset {dataset.__class__.__name__} using annotation type {self.ann_type.name}')
        if self.ann_type is AnnotationTypes.SPECTRAL_DINO_SEG_8 or self.ann_type is AnnotationTypes.SPECTRAL_DINO_SEG_16:
            if segment_foreground is None:
                raise ImportError('Foreground segmentation is not available. Please install the requirements for foreground segmentation.')
            return segment_foreground(dataset_ann_path, dataset, self.ann_type,
                                      self.batch_size, self.device, self.num_workers, self.disable_tqdm)
        elif self.ann_type is AnnotationTypes.DINO_SEG_8 or self.ann_type is AnnotationTypes.DINO_SEG_16:
            return self._get_obj_boxes(dataset_ann_path, dataset)
        else:
            raise NotImplementedError(f'Annotation type {self.ann_type.name} not implemented')

    def _get_obj_boxes(self, dataset_ann_path, dataset):
        obj_box_file_name = self.OBJ_BOX_FILE_PREFIX + f'_{self.ann_model.patch_size}_{self.dino_att_threshold}_{self.att_head_threshold}.pt'
        obj_box_file_path = os.path.join(dataset_ann_path, obj_box_file_name)
        if os.path.isfile(obj_box_file_path):
            LOGGER.info(f'Using object boxes saved in {obj_box_file_path}')
            with open(obj_box_file_path, 'rb') as f:
                obj_boxes = torch.load(f, map_location='cpu')
            return obj_boxes

        # backup dataset transform, which is modified during box prediction
        dataset_trans_backup = dataset.transform

        trans, resize_img_size = get_resize_and_cropping_transforms(self.target_img_size, cropping=False)
        if self.ann_model.normalise_transform is not None:
            trans += [transforms.Normalize(self.ann_model.normalise_transform.mean, self.ann_model.normalise_transform.std)]
        dataset.transform = transforms.Compose(trans)

        obj_boxes = torch.zeros((len(dataset), 4), device=self.device)

        w_featmap = self.target_img_size // self.ann_model.patch_size

        img_data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        for batch_idx, (imgs, _) in enumerate(tqdm(img_data_loader, leave=False, desc=f'Predicting boxes for {dataset.__class__.__name__}', disable=self.disable_tqdm)):

            batch_offset = batch_idx * self.batch_size
            imgs = imgs.to(self.device, non_blocking=True)

            _, th_attn_agg_heads = self._get_filtered_attention(imgs, w_featmap)

            obj_boxes[batch_offset:batch_offset + imgs.shape[0]] = self._get_box_from_attention(
                th_attn_agg_heads, w_featmap)

        with open(obj_box_file_path, 'wb') as f:
            torch.save(obj_boxes, f)
        LOGGER.debug(f'Saved object boxes in {obj_box_file_path}')

        obj_boxes = obj_boxes.to('cpu')

        dataset.transform = dataset_trans_backup
        return obj_boxes

    def _get_filtered_attention(self, imgs, w_featmap):
        # determine object box based on self attention map
        # based on https://github.com/facebookresearch/dino/blob/main/visualize_attention.py
        att = self.ann_model.model.get_last_selfattention(imgs)
        nh = att.shape[1]
        t = att[:, :, 0, 1:].reshape(imgs.shape[0], nh, -1)

        # only keep a certain percentage of the mass
        val, idx = torch.sort(t)
        val /= torch.sum(val, dim=-1, keepdim=True)
        cumval = torch.cumsum(val, dim=-1)
        th_attn = cumval > 1 - self.dino_att_threshold
        idx2 = torch.argsort(idx)

        # for i in range(imgs.shape[0]):
        #     for head in range(nh):
        #         th_attn[i, head] = th_attn[i, head][idx2[i, head]]
        img_idxs = torch.arange(imgs.shape[0]).repeat_interleave(nh)
        head_idxs = torch.arange(nh).repeat(imgs.shape[0])
        th_attn[img_idxs, head_idxs] = th_attn[img_idxs, head_idxs][
            torch.arange(img_idxs.shape[0]).view(-1, 1), idx2[img_idxs, head_idxs]]

        th_attn = th_attn.reshape(imgs.shape[0], nh, w_featmap, w_featmap).float()
        th_attn_agg_heads = th_attn.sum(1) > self.att_head_threshold
        return th_attn, th_attn_agg_heads

    def _get_box_from_attention(self, att, w_featmap):
        boxes = torch.zeros(att.shape[0], 4)
        for i in range(att.shape[0]):
            patch_coord = get_box_from_binary_mask(att[i])
            boxes[i] = patch_coord / w_featmap
        return boxes

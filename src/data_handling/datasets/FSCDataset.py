import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import PILToTensor

from .CountingDataset import CountingDataset
from ..DatasetSplits import DatasetSplits
from ...util import plotting

LOGGER = logging.getLogger()


class FSCDataset(CountingDataset):
    META_FILE = 'imgIdx.mat'
    INVALID_CLS = 0

    def __init__(self, base_dir, use_133_subset=False, plot_dir=None, in_memory=True, num_shots=1,
                 split=DatasetSplits.TEST, min_count=None, max_count=None, density_scaling=1.0, **kwargs):
        self.use_133_subset = use_133_subset
        self.density_scaling = density_scaling

        self.min_count = min_count  # inclusive
        self.max_count = max_count  # exclusive

        super().__init__(base_dir, 5, split=split, img_sub_dir='images_384_VarV2', plot_dir=plot_dir, in_memory=in_memory, dataset_sub_dir='FSC147_384_V2', **kwargs)

        self.num_shots = num_shots
        if num_shots < 0 or num_shots > 3:
            raise ValueError(f'Number of shots must be between 0 and 3, but is {num_shots}')

        self._prepare_dataset()

    def reduce_to_subset(self, idxs):

        self.ref_boxes = self.ref_boxes[idxs]
        self.gt_density_maps = [self.gt_density_maps[i] for i in idxs]
        self.split_classes = [self.split_classes[i] for i in idxs]
        self.file_names = [self.file_names[i] for i in idxs]
        if self.dataset is not None:
            self.dataset = [self.dataset[i] for i in idxs]

        self.length = len(self.file_names)

    def _get_annotation_file_paths(self):
        if self.use_133_subset:
            dataset_num = 133
        else:
            dataset_num = 147
        return f'annotation_FSC{dataset_num}_384.json', f'ImageClasses_FSC{dataset_num}.txt', f'Train_Test_Val_FSC_{dataset_num}.json'

    def __getitem__(self, index: int):
        img, target_dict = super().__getitem__(index)

        target_dict[self.LABEL_DICT_BOXES] = self.ref_boxes[index]

        target_dict[self.LABEL_DICT_DENSITY_MAPS] = self.gt_density_maps[index]
        target_dict[self.LABEL_DICT_ALL_DENSITY_MAPS] = self.gt_density_maps[index]

        target_dict[self.LABEL_DICT_CLASSES] = self.split_classes[index]

        if self.transform is not None:
            img, target_dict = self.transform(img, target_dict)

        if self.use_reference_crops:
            # save rescaled image crops as references
            self._add_ref_imgs(img, target_dict, self.ref_boxes.shape[1])
        return img, target_dict

    def _get_relative_box_coords(self, image, boxes, max_hw=384):
        W, H = image.size
        if W > max_hw or H > max_hw:
            scale_factor = float(max_hw) / max(H, W)
            new_H = 8 * int(H * scale_factor / 8)
            new_W = 8 * int(W * scale_factor / 8)
            from src.util import transforms
            resized_image = transforms.Resize((new_H, new_W))(image)
        else:
            scale_factor = 1
            resized_image = image

        lines_boxes = list()
        for bbox in boxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            lines_boxes.append([x1, y1, x2, y2])

        boxes = torch.tensor(lines_boxes)
        boxes = boxes * scale_factor
        boxes = boxes.float()
        boxes[:, [0, 2]] /= resized_image.width
        boxes[:, [1, 3]] /= resized_image.height
        return boxes

    def _setup(self):
        ann_file_name, cls_file_name, split_file_name = self._get_annotation_file_paths()
        with open(os.path.join(self.root, ann_file_name), 'r') as f:
            ann_file = json.load(f)
        with open(os.path.join(self.root, cls_file_name), 'r') as f:
            self.raw_cls_labels = f.readlines()
            self.img_name_2_cls_name = {c.split('\t')[0]: c.split('\t')[1][:-1] for c in self.raw_cls_labels}
            self.idx_to_class = sorted(set(self.img_name_2_cls_name.values()))
            self.class_to_idx = {c: i for i, c in enumerate(self.idx_to_class)}

        with open(os.path.join(self.root, split_file_name), 'r') as f:
            self.split_file = json.load(f)
            self.split_file_names = self.split_file[self.split.name.lower()]

        self.split_annotations = [ann_file[img_name] for img_name in self.split_file_names]
        self.split_classes = [self.class_to_idx[self.img_name_2_cls_name[img_name]] for img_name in self.split_file_names]
        self.gt_density_maps = []
        for img_name in self.split_file_names:
            gt_density_map = torch.from_numpy(np.load(
                os.path.join(self.root, 'gt_density_map_adaptive_384_VarV2', img_name.rsplit('.', 1)[0] + '.npy')))
            gt_density_map *= self.density_scaling
            self.gt_density_maps.append(gt_density_map)
        self.labels = torch.tensor([len(ann['points']) for ann in self.split_annotations])

        # filter out images with too few or too many objects
        sample_mask = torch.ones(len(self.split_annotations), dtype=torch.bool)
        if self.min_count is not None:
            sample_mask &= self.labels >= self.min_count
        if self.max_count is not None:
            sample_mask &= self.labels < self.max_count
        self.split_file_names = [self.split_file_names[i] for i in range(len(self.split_file_names)) if sample_mask[i]]
        self.split_annotations = [self.split_annotations[i] for i in range(len(self.split_annotations)) if sample_mask[i]]
        self.split_classes = [self.split_classes[i] for i in range(len(self.split_classes)) if sample_mask[i]]
        self.gt_density_maps = [self.gt_density_maps[i] for i in range(len(self.gt_density_maps)) if sample_mask[i]]
        self.labels = self.labels[sample_mask]

        self.cls_idx_2_img_idxs = {c_idx: [] for c_idx in range(len(self.idx_to_class))}
        for img_idx, img_name in enumerate(self.split_file_names):
            cls_idx = self.class_to_idx[self.img_name_2_cls_name[img_name]]
            self.cls_idx_2_img_idxs[cls_idx].append(img_idx)

    def _prepare_dataset(self):
        # in xyxy format
        self.ref_boxes = []

        for i in range(len(self.split_annotations)):
            image = self._load_image(i)
            boxes = self.split_annotations[i]['box_examples_coordinates']
            boxes = self._get_relative_box_coords(image, boxes)
            self.ref_boxes.append(boxes)

        self.ref_boxes = self.pad_all_samples(self.ref_boxes)

    def _get_file_names(self):
        return self.split_file_names

    def _get_labels(self):
        return self.labels

    def visualise_sample(self, index, show_boxes=True,
                         save_figure=False, show_figure=True, use_orig_image=False):
        img, target_dict = self[index]
        label = target_dict[self.LABEL_DICT_GLOBAL_COUNT]
        boxes = target_dict[self.LABEL_DICT_BOXES]
        if use_orig_image:
            img = PILToTensor()(self._load_image(index))
        boxes = FSCDataset.unpad_sample(boxes)
        scaled_boxes = boxes.clone()
        scaled_boxes[:, [0, 2]] *= img.shape[2]
        scaled_boxes[:, [1, 3]] *= img.shape[1]
        fig = plt.figure()
        title = f'{int(label)} salient object(s)'
        plt.imshow(img.permute((1, 2, 0)))

        # plot annotated boxes
        if show_boxes:
            title += f', {len(boxes)} reference boxes'
            plotting.plot_boxes(scaled_boxes, fig)

        plt.title(title)

        plt.axis('off')
        plt.tight_layout()
        if save_figure:
            plt.savefig(os.path.join(self.plot_dir, f'sample_{index}.png'))
        if show_figure:
            plt.show()
        else:
            return fig, img

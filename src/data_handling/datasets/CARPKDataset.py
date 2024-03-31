import logging
import os

import numpy as np
import torch
from PIL import Image

from .CountingDataset import CountingDataset
from ..DatasetSplits import DatasetSplits

LOGGER = logging.getLogger()


class CARPKDataset(CountingDataset):
    META_FILE = 'imgIdx.mat'
    INVALID_CLS = 0

    def __init__(self, base_dir, plot_dir=None, in_memory=True, num_shots=3, **kwargs):
        super().__init__(base_dir, 5, split=DatasetSplits.TEST, img_sub_dir='Images', plot_dir=plot_dir, in_memory=in_memory,
                         **kwargs)
        self.num_shots = num_shots
        self.idx_to_class = self.id_to_class
        self.class_labels = None
        self.boxes = None

    def _setup(self):

        with open(os.path.join(self.root, 'ImageSets', 'test.txt'), 'r') as f:
            self.imgs = [l[:-1] + '.png' for l in f.readlines()]
        self.annotations = {}
        bboxes = []
        for im_id in self.imgs:
            image = Image.open(os.path.join(self.img_dir, im_id))
            W, H = image.size
            with open(os.path.join(self.root, 'Annotations', f'{im_id.rsplit(".", 1)[0]}.txt'), 'r') as f:
                img_ann = {'box_examples_coordinates': [], 'points': []}
                for obj in f.readlines():
                    split_obj = obj.split(' ')
                    box = np.array(split_obj[:4], dtype=int).reshape(2, 2)
                    img_ann['box_examples_coordinates'].append(box)
                    img_ann['points'].append(box[0] + box[1] / 2)
                img_ann['box_examples_coordinates'] = np.stack(img_ann['box_examples_coordinates'])
                img_ann['points'] = np.stack(img_ann['points'])

                boxes = []
                for bbox in img_ann['box_examples_coordinates']:
                    x1 = bbox[0][0] / W
                    y1 = bbox[0][1] / H
                    x2 = bbox[1][0] / W
                    y2 = bbox[1][1] / H
                    boxes.append([x1, y1, x2, y2])

                self.annotations[im_id] = img_ann
            bboxes.append(torch.tensor(boxes))
        self.bboxes = CARPKDataset.pad_all_samples(bboxes)

    def _get_file_names(self):
        return self.imgs

    def _get_labels(self):
        labels = []
        for im_id in self.imgs:
            labels.append(len(self.annotations[im_id]['points']))
        return torch.tensor(labels)

    def __getitem__(self, index: int):
        img, target_dict = super().__getitem__(index)

        target_dict[self.LABEL_DICT_BOXES] = self.bboxes[index]

        if self.transform is not None:
            img, target_dict = self.transform(img, target_dict)

        if self.use_reference_crops:
            # save rescaled image crops as references
            self._add_ref_imgs(img, target_dict, self.bboxes.shape[1])
        return img, target_dict

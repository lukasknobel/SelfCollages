import logging
import os
import time
from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from tqdm import tqdm

from ..DatasetSplits import DatasetSplits
from ...util import Constants, transforms

LOGGER = logging.getLogger()


class CountingDataset(VisionDataset, ABC):
    PAD_VAL = -1

    LABEL_DICT_GLOBAL_COUNT = Constants.TARGET_DICT_GLOBAL_COUNT
    LABEL_DICT_TOTAL_NUM_OBJECTS = Constants.TARGET_DICT_TOTAL_NUM_OBJECTS  # total number of objects (used in the reference setting where this includes objects from all clusters)
    LABEL_DICT_BOXES = Constants.TARGET_DICT_BOXES
    LABEL_DICT_ALL_BOXES = Constants.TARGET_DICT_ALL_BOXES
    LABEL_DICT_DENSITY_MAPS = Constants.TARGET_DICT_DENSITY_MAPS
    LABEL_DICT_ALL_DENSITY_MAPS = Constants.TARGET_DICT_ALL_DENSITY_MAPS
    LABEL_DICT_CLASSES = Constants.TARGET_DICT_CLASSES
    LABEL_DICT_NUM_ELEMENTS = Constants.TARGET_DICT_NUM_ELEMENTS
    LABEL_DICT_REF_IMGS = Constants.TARGET_DICT_REF_IMGS
    LABEL_DICT_IS_ZERO_SHOT = Constants.TARGET_DICT_IS_ZERO_SHOT

    NUM_OBJ_CLASSES = 91

    def __init__(self, base_dir, num_classes=21, split=DatasetSplits.TRAIN, dataset_sub_dir=None, img_sub_dir=None,
                 plot_dir=None, in_memory=False, transform=None, enable_annotation=False,
                 ann_threshold=0.9, disable_tqdm=False, reference_crop_size=None,
                 use_reference_crops=False, **kwargs):
        logging.info(f'Creating {self.__class__.__name__}')
        if dataset_sub_dir is None:
            dataset_sub_dir = self.get_default_data_sub_dir()
        self.base_dir = base_dir
        root = os.path.join(base_dir, dataset_sub_dir)
        super().__init__(root, self._load_image)
        if len(kwargs) > 0:
            logging.info(f'The following arguments are ignored: {kwargs}')
        self.num_classes = num_classes
        self.transform = transform
        self.split = split
        self.in_memory = in_memory
        self.disable_tqdm = disable_tqdm

        self.use_reference_crops = use_reference_crops

        self.ann_box_file = f'ann_box_{split.name.lower()}.pt'
        self.ann_labels_file = f'ann_labels_{split.name.lower()}.pt'
        self.ann_scores_file = f'ann_scores_{split.name.lower()}.pt'

        self.pred_box_path = os.path.join(self.root, self.ann_box_file)
        self.pred_label_path = os.path.join(self.root, self.ann_labels_file)
        self.pred_score_path = os.path.join(self.root, self.ann_scores_file)

        self.reference_crop_size = reference_crop_size
        self.ref_crop_transform = None
        if self.reference_crop_size is not None:
            self.ref_crop_transform = transforms.Resize((self.reference_crop_size, self.reference_crop_size))

        if not os.path.isdir(self.root):
            os.makedirs(self.root)

        if plot_dir is None:
            self.plot_dir = os.path.join(self.root, self.__class__.__name__)
        else:
            self.plot_dir = os.path.join(plot_dir, self.__class__.__name__)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        if img_sub_dir is not None:
            self.img_dir = os.path.join(self.root, img_sub_dir)
        else:
            self.img_dir = self.root

        # annotation model
        if enable_annotation:
            self.ann_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.ann_model = fasterrcnn_resnet50_fpn_v2(weights=self.ann_weights, box_score_thresh=ann_threshold)
            self.ann_model.eval()
            num_ann_classes = self.ann_model.roi_heads.box_predictor.cls_score.out_features
            if num_ann_classes != CountingDataset.NUM_OBJ_CLASSES:
                raise ValueError(
                    f'Number of object classes ({num_ann_classes}) different from the predefined number ({CountingDataset.NUM_OBJ_CLASSES}).')
        else:
            self.ann_model = None

        self._setup()

        self.file_names = self._get_file_names()
        self.length = len(self.file_names)

        self.id_to_class = self._get_id_to_class()

        self.dataset = None
        if self.in_memory:
            # if in_memory, load all images
            self.dataset = self._load_imgs()

        self.labels = self._get_labels()

    @classmethod
    def get_default_data_sub_dir(cls):
        return cls.__name__[:-7]

    @staticmethod
    def pad_all_samples(samples):
        """
        Pads a list of samples to construct a single tensor
        """
        max_num_elements, elements_dim = \
            torch.tensor([[b.shape[0], b.shape[1]] if len(b.shape) > 1 else [b.shape[0], 0] for b in samples]).max(dim=0)[0]

        if elements_dim == 0:
            padded_samples = torch.ones((len(samples), max_num_elements), dtype=samples[0].dtype) * CountingDataset.PAD_VAL
        else:
            padded_samples = torch.ones((len(samples), max_num_elements, elements_dim), dtype=samples[0].dtype) * CountingDataset.PAD_VAL
        for i, samples in enumerate(samples):
            if samples.shape[0] > 0:
                padded_samples[i, :samples.shape[0]] = samples
        return padded_samples

    @staticmethod
    def pad_sample(sample, max_num_elements, elements_dim, dtype):
        """
        Pads pads a single tensor to the specified dimensions
        """
        if isinstance(elements_dim, tuple):
            padded_samples = torch.ones((max_num_elements, *elements_dim), dtype=dtype) * CountingDataset.PAD_VAL
            padded_samples[:sample.shape[0]] = sample
        elif elements_dim == 0:
            padded_samples = torch.ones((max_num_elements,), dtype=dtype) * CountingDataset.PAD_VAL
            padded_samples[:sample.shape[0]] = sample
        else:
            padded_samples = torch.ones((max_num_elements, elements_dim), dtype=dtype) * CountingDataset.PAD_VAL
            if sample.shape[1] != elements_dim:
                raise ValueError(f'Elements dimension ({sample.shape[1]}) does not match the specified dimension ({elements_dim}) for padding.')
            padded_samples[:sample.shape[0]] = sample

        return padded_samples

    @staticmethod
    def unpad_sample(sample):
        num_elements = (sample == CountingDataset.PAD_VAL).view(len(sample), -1).any(dim=1).int().nonzero()
        if len(num_elements) == 0:
            return sample
        else:
            return sample[:num_elements[0]]

    @staticmethod
    def modify_few_shot_target_dict(target_dict, shot_num):
        zero_shot_mask = torch.zeros_like(target_dict[CountingDataset.LABEL_DICT_GLOBAL_COUNT], dtype=torch.bool)
        if isinstance(shot_num, torch.Tensor):
            zero_shot_mask = shot_num == 0
        elif shot_num == 0:
            zero_shot_mask = torch.ones_like(target_dict[CountingDataset.LABEL_DICT_GLOBAL_COUNT], dtype=torch.bool)

        # for 0 shots, the total number of objects is the same as the global count
        target_dict[CountingDataset.LABEL_DICT_GLOBAL_COUNT][zero_shot_mask] = target_dict[CountingDataset.LABEL_DICT_TOTAL_NUM_OBJECTS][zero_shot_mask]
        if CountingDataset.LABEL_DICT_ALL_BOXES in target_dict:
            target_dict[CountingDataset.LABEL_DICT_BOXES][zero_shot_mask] = target_dict[CountingDataset.LABEL_DICT_ALL_BOXES][zero_shot_mask]
        if CountingDataset.LABEL_DICT_ALL_DENSITY_MAPS in target_dict:
            target_dict[CountingDataset.LABEL_DICT_DENSITY_MAPS][zero_shot_mask] = target_dict[CountingDataset.LABEL_DICT_ALL_DENSITY_MAPS][zero_shot_mask]

    def __getitem__(self, index: int):
        if self.in_memory:
            img = self.dataset[index]
        else:
            img = self._load_image(index)
        target_dict = {
            CountingDataset.LABEL_DICT_GLOBAL_COUNT: self.labels[index],
            # by default, the total number of objects is the same as the global count
            CountingDataset.LABEL_DICT_TOTAL_NUM_OBJECTS: self.labels[index]
        }

        return img, target_dict

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def _get_file_names(self):
        pass

    @abstractmethod
    def _get_labels(self):
        pass

    def measure_dataset_indexing_time(self, num_samples=10, disable_tqdm=False):
        gen_times = []
        rand_idxs = torch.randint(len(self), (num_samples,))
        for rand_idx in tqdm(rand_idxs, disable=disable_tqdm):
            start = time.time()
            self[rand_idx]
            gen_times.append(time.time() - start)
        gen_times = torch.tensor(gen_times)
        LOGGER.info(f'Generation time: {gen_times.mean().item(): .4f}+-{gen_times.std(): .4f}s')

    def _get_id_to_class(self):
        if self.ann_model is not None:
            return self.ann_weights.meta["categories"]
        else:
            return None

    def _add_ref_imgs(self, img, target_dict, max_num_references):
        ref_boxes = self.unpad_sample(target_dict[self.LABEL_DICT_BOXES])
        ref_boxes = ref_boxes[:max_num_references]
        scaled_ref_boxes = ref_boxes.clone()
        scaled_ref_boxes[:, [0, 2]] *= img.shape[2]
        scaled_ref_boxes[:, [1, 3]] *= img.shape[1]
        scaled_ref_boxes = scaled_ref_boxes.to(torch.int)
        single_ref_dim = (img.shape[0], self.reference_crop_size, self.reference_crop_size)
        ref_crops = torch.zeros((ref_boxes.shape[0], *single_ref_dim), dtype=img.dtype)
        for i, scaled_ref_box in enumerate(scaled_ref_boxes):
            ref_crop = img[:, scaled_ref_box[1]:scaled_ref_box[3], scaled_ref_box[0]:scaled_ref_box[2]]
            ref_crops[i] = self.ref_crop_transform(ref_crop)
        target_dict[self.LABEL_DICT_REF_IMGS] = self.pad_sample(ref_crops, max_num_elements=max_num_references,
                                                                elements_dim=single_ref_dim, dtype=torch.float32)

    @torch.no_grad()
    def _predict_boxes(self, index=None, batch_size=16):
        """
        Based on https://pytorch.org/vision/stable/models.html#object-detection
        :param index:
        :return:
        """

        if self.ann_model is None:
            LOGGER.error(f'you have to enable annotation by passing enable_annotation=True when creating the dataset to use _predict_boxes')
            return None, None
        preprocess = self.ann_weights.transforms()

        if index is None:
            if os.path.isfile(self.pred_box_path) and os.path.isfile(self.pred_label_path) and os.path.isfile(self.pred_score_path):
                LOGGER.info('Returning saved box predictions')
                pred_boxes = torch.load(self.pred_box_path)
                pred_labels = torch.load(self.pred_label_path)
                pred_scores = torch.load(self.pred_score_path)
                return pred_boxes, pred_labels, pred_scores
            imgs = [[self[i][0] for i in range(b * batch_size, (b + 1) * batch_size) if i < len(self)] for b in
                    range(int(np.ceil(len(self) / batch_size)))]
        else:
            imgs = [[self[index][0]]]

        LOGGER.info(f'Predicting boxes for {len(self)} images')
        pred_boxes = []
        pred_labels = []
        pred_scores = []

        for batch_imgs in tqdm(imgs, disable=self.disable_tqdm or len(imgs) < 2):
            batch = [preprocess(img) for img in batch_imgs]

            prediction = self.ann_model(batch)
            for img, pred in zip(batch_imgs, prediction):
                w = img.shape[2]
                h = img.shape[1]
                box = pred['boxes'].cpu()
                box[:, [0, 2]] /= w
                box[:, [1, 3]] /= h
                pred_boxes.append(box)
                pred_labels.append(pred['labels'].cpu())
                pred_scores.append(pred['scores'].cpu())

        if index is None:
            with open(self.pred_box_path, 'wb') as f:
                torch.save(pred_boxes, f)
            with open(self.pred_label_path, 'wb') as f:
                torch.save(pred_labels, f)
            with open(self.pred_score_path, 'wb') as f:
                torch.save(pred_scores, f)
        else:
            pred_boxes = pred_boxes[0]
            pred_labels = pred_labels[0]
            pred_scores = pred_scores[0]

        return pred_boxes, pred_labels, pred_scores

    def _setup(self):
        return

    def _get_img_path(self, index):
        return os.path.join(self.img_dir, self.file_names[index])

    def _load_image(self, index):
        return Image.open(self._get_img_path(index)).convert("RGB")

    def _load_imgs(self):
        logging.debug('Loading images')
        imgs = []
        for i in range(len(self)):
            img = self._load_image(i)
            imgs.append(img)
        return imgs

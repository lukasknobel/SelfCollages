import logging
import os

import mat73
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from torchvision.transforms import PILToTensor

from .CountingDataset import CountingDataset
from ..DatasetSplits import DatasetSplits
from ...util import plotting
from ...util.misc import compute_intersection_and_union

LOGGER = logging.getLogger()


class MSODataset(CountingDataset):
    META_FILE = 'imgIdx.mat'
    INVALID_CLS = 0

    def __init__(self, base_dir, plot_dir=None, visualise=False, in_memory=True, split=DatasetSplits.TEST,
                 few_shot=False, num_shots=1, **kwargs):
        super().__init__(base_dir, 5, split=split, img_sub_dir='img', plot_dir=plot_dir, in_memory=in_memory,
                         enable_annotation=True, **kwargs)

        self.visualise = visualise
        self.idx_to_class = self.id_to_class
        self.class_labels = None
        self.boxes = None

        self._prepare_dataset()

        if few_shot:
            self.use_reference_crops = True
            self.num_shots = num_shots
            # use actual labels
            if self.boxes is not None:
                self.labels = torch.tensor([self.unpad_sample(b).shape[0] for b in self.boxes])
            self._filter_min_objs(self.num_shots)

        else:
            self.num_shots = 0

    def _filter_min_objs(self, min_objs):
        mask = self.labels >= min_objs

        self.length = mask.sum()

        self.labels = self.labels[mask]
        if self.boxes is not None:
            self.boxes = self.boxes[mask]
        if self.class_labels is not None:
            self.class_labels = self.class_labels[mask]

        if self.dataset is not None:
            self.dataset = [self.dataset[i] for i in range(len(self.dataset)) if mask[i]]


    @staticmethod
    def _match_boxes(boxes_a, boxes_b, thresh=0.7):
        ious = torch.zeros((boxes_a.shape[0], boxes_b.shape[0]))
        matched_boxes = -torch.ones(boxes_a.shape[0], dtype=torch.int64)
        matched_a = torch.zeros(boxes_a.shape[0], dtype=torch.bool)
        matched_b = torch.zeros(boxes_b.shape[0], dtype=torch.bool)
        if boxes_b.shape[0] > 0:
            for i, box_a in enumerate(boxes_a):
                for j, box_b in enumerate(boxes_b):
                    intersection, union = compute_intersection_and_union(box_a, box_b)
                    ious[i, j] = intersection / union
                sorted_ious = torch.argsort(ious[i], descending=True)
                # discard boxes in b that were already matched
                available_boxes = sorted_ious[~matched_b[sorted_ious]]
                if len(available_boxes) > 0 and ious[i, available_boxes[0]] >= thresh:
                    matched_a[i] = True
                    matched_b[available_boxes[0]] = True
                    matched_boxes[i] = available_boxes[0]

        return matched_boxes, matched_a, matched_b, ious

    def __getitem__(self, index: int):
        img, target_dict = super().__getitem__(index)
        if self.boxes is not None and self.class_labels is not None:
            target_dict[self.LABEL_DICT_BOXES] = self.boxes[index]
            target_dict[self.LABEL_DICT_CLASSES] = self.class_labels[index]

            obj_classes = self.unpad_sample(target_dict[self.LABEL_DICT_CLASSES])
            target_dict[self.LABEL_DICT_NUM_ELEMENTS] = obj_classes.shape[0]
        if self.transform is not None:
            img, target_dict = self.transform(img, target_dict)

        if self.boxes is not None and self.class_labels is not None and self.use_reference_crops:
            # save rescaled image crops as references
            self._add_ref_imgs(img, target_dict, self.boxes.shape[1])
        return img, target_dict

    def _setup(self):
        self.meta_file = mat73.loadmat(os.path.join(self.root, MSODataset.META_FILE))['imgIdx']

    def _get_file_names(self):
        return self.meta_file['name']

    def _get_labels(self):
        return torch.tensor(np.stack(self.meta_file['label']), dtype=torch.long)

    def _prepare_dataset(self):
        # read meta file
        if self.split is not DatasetSplits.TEST:
            raise ValueError(f'Split {self.split.name} is not supported')
        logging.debug('Loading meta data')
        self.boxes = []
        for box in self.meta_file['anno']:
            if len(box.shape) == 1:
                if box.shape[0] == 2:
                    self.boxes.append(torch.tensor([], dtype=torch.float32))
                else:
                    self.boxes.append(torch.tensor(box, dtype=torch.float32).view(1, -1))
            else:
                self.boxes.append(torch.tensor(box, dtype=torch.float32))
        # normalise box coordinates
        for i in range(len(self)):
            box = self.boxes[i]
            if box.shape[0] > 0:
                img = self._load_image(i)
                box[:, [0, 2]] /= img.width
                box[:, [1, 3]] /= img.height
            # clip boxes to fix faulty annotations
            self.boxes[i] = box.clip(0)

        self.boxes = MSODataset.pad_all_samples(self.boxes)
        # get object classes
        self.all_boxes = None
        self.all_class_labels = None
        self.orig_annotated = None
        logging.debug('Predicting object boxes and classes')
        self.all_boxes, self.all_class_labels, self.orig_annotated = self._create_box_class_annotations()
        filtered_class_labels = [sample_class_labels[sample_orig_annotated] for sample_class_labels, sample_orig_annotated in zip(self.all_class_labels, self.orig_annotated)]
        self.class_labels = MSODataset.pad_all_samples(filtered_class_labels).to(torch.int64)
        # get prediction statistics
        # object class statistics
        num_obj_classes_orig = np.unique([len(set([int(c) for c, a in zip(classes, annotations) if a and int(c)!=MSODataset.INVALID_CLS])) for classes, annotations in zip(self.all_class_labels, self.orig_annotated) if annotations.any()], return_counts=True)
        upper_bound_num_obj_classes_orig = np.unique([len(set([int(c) for c, a in zip(classes, annotations) if a and int(c)!=MSODataset.INVALID_CLS])) + sum([1 for c, a in zip(classes, annotations) if a and int(c)==MSODataset.INVALID_CLS]) for classes, annotations in zip(self.all_class_labels, self.orig_annotated) if annotations.any()], return_counts=True)
        LOGGER.info(f'Number of object classes in samples with at least one annotated object: {num_obj_classes_orig[0]}, {num_obj_classes_orig[1]}')
        LOGGER.info(f'Upper bound of the number of classes (if all unknown classes were different): {upper_bound_num_obj_classes_orig[0]}, {upper_bound_num_obj_classes_orig[1]}')
        # label statistics
        are_all_boxes_labeled = torch.stack([(class_labels != MSODataset.INVALID_CLS)[orig_annotated].all() for class_labels, orig_annotated in
                                             zip(self.all_class_labels, self.orig_annotated)])
        num_fully_labeled_images_per_count = [int(sum(a)) for a in np.split(are_all_boxes_labeled, np.unique(self.labels, return_index=True)[1])[1:]]
        LOGGER.info(f'Number of fully labeled samples per count: {num_fully_labeled_images_per_count} (In total: {sum(num_fully_labeled_images_per_count)} out of {self.labels.shape[0]})')

        # box statistics
        total_num_boxes = torch.concat([a for a in self.orig_annotated if a.shape[0] > 0]).shape[0]
        num_annotated_boxes = torch.arange(total_num_boxes)[torch.concat([a for a in self.orig_annotated if a.shape[0] > 0])].shape[0]
        total_matched_boxes = torch.sum((torch.concat([l for l in self.all_class_labels if l.shape[0] > 0]) != MSODataset.INVALID_CLS) & torch.concat([a for a in self.orig_annotated if a.shape[0] > 0]))
        num_unk_classes = int(torch.sum(torch.concat([l for l in self.all_class_labels if l.shape[0] > 0]) == MSODataset.INVALID_CLS))
        total_add_boxes = int(torch.sum(torch.concat([~a for a in self.orig_annotated if a.shape[0] > 0])))
        add_boxes_for_nonempty = int(torch.sum(torch.concat([~a for a, l in zip(self.orig_annotated, self.labels) if a.shape[0] > 0 and l > 0])))
        LOGGER.info(
            f'{total_num_boxes} boxes in total, '
            f'{num_annotated_boxes} annotated boxes, '
            f'{total_matched_boxes} matched boxes, '
            f'{num_unk_classes} annotated boxes of unknown class, '
            f'{total_add_boxes} additional predicted boxes, '
            f'{add_boxes_for_nonempty} additional predicted boxes in images with at least one annotated object'
        )

        if self.visualise:
            self.visualise_sample(1001)
            # sample with different types of objects and between 1-3 annotated objects
            for idx in [i for i in range(len(self)) if
                        1 < torch.unique(self.all_class_labels[i]).shape[0] and 0 < self.labels[i] < 4][-10:]:
                self.visualise_sample(idx)
            # all samples with 4+ annotated objects
            for i in torch.arange(len(self))[self.labels == 4]:
                self.visualise_sample(i)
            # one sample of each of the labelled object counts
            for i in np.unique(self.labels, return_index=True)[1]:
                self.visualise_sample(i)

    def _create_box_class_annotations(self):
        all_pred_boxes, all_pred_classes, _ = self._predict_boxes()
        boxes = []
        class_labels = []
        orig_annotated = []
        for i, sample_ann_boxes in enumerate(self.boxes):
            pred_boxes = all_pred_boxes[i]
            pred_classes = all_pred_classes[i]

            # match annotated and predicted boxes
            sample_ann_boxes = MSODataset.unpad_sample(sample_ann_boxes)
            match_idxs, matched_ann, matched_pred, ious = MSODataset._match_boxes(sample_ann_boxes, pred_boxes)

            # prepare data for predicted boxes
            unmatched_pred_boxes = pred_boxes[~matched_pred]
            unmatched_orig_annotated = torch.zeros(unmatched_pred_boxes.shape[0], dtype=torch.bool)
            unmatched_class_labels = pred_classes[~matched_pred]

            if sample_ann_boxes.shape[0] > 0:
                # add annotated boxes
                sample_boxes = sample_ann_boxes
                sample_orig_annotated = torch.ones(sample_ann_boxes.shape[0], dtype=torch.bool)
                # get predicted classes for annotated boxes
                sample_class_labels = torch.ones(sample_ann_boxes.shape[0], dtype=torch.int64) * MSODataset.INVALID_CLS
                if len(pred_classes) > 0:
                    sample_class_labels[matched_ann] = pred_classes[match_idxs][matched_ann]

                # add predicted boxes
                sample_boxes = torch.concatenate([sample_boxes, unmatched_pred_boxes])
                sample_orig_annotated = torch.concatenate([sample_orig_annotated, unmatched_orig_annotated])
                sample_class_labels = torch.concatenate([sample_class_labels, unmatched_class_labels])
            else:
                # Use only predicted boxes
                sample_boxes = unmatched_pred_boxes
                sample_orig_annotated = unmatched_orig_annotated
                sample_class_labels = unmatched_class_labels

            boxes.append(sample_boxes)
            class_labels.append(sample_class_labels)
            orig_annotated.append(sample_orig_annotated)

        return boxes, class_labels, orig_annotated

    def visualise_sample(self, index, show_annotated_boxes=True, show_box_predictions=True, show_box_matches=False,
                         show_intersection=True, save_figure=True, show_figure=True, use_orig_image=False):
        img, target_dict = self[index]
        label = target_dict[self.LABEL_DICT_GLOBAL_COUNT]
        boxes = target_dict[self.LABEL_DICT_BOXES]
        if use_orig_image:
            img = PILToTensor()(self._load_image(index))
        boxes = MSODataset.unpad_sample(boxes)
        scaled_boxes = boxes.clone()
        scaled_boxes[:, [0, 2]] *= img.shape[2]
        scaled_boxes[:, [1, 3]] *= img.shape[1]
        fig = plt.figure()
        title = f'{int(label)} salient object(s)'
        plt.imshow(img.permute((1, 2, 0)))

        # plot annotated boxes
        if show_annotated_boxes:
            title += f', {len(boxes)} annotated boxes'
            plotting.plot_boxes(scaled_boxes, fig)

        # plot predicted boxes and labels
        if show_box_predictions or show_box_matches:
            if not use_orig_image:
                LOGGER.warning(f'Predicted boxes are shown based on the original image and may not be aligned when using cropping/flipping for visualisation')
            pred_boxes, pred_class_ids, _ = self._predict_boxes(index)

            title += f', {pred_boxes.shape[0]} predicted boxes'
            scaled_pred_boxes = pred_boxes.clone()
            scaled_pred_boxes[:, [0, 2]] *= img.shape[2]
            scaled_pred_boxes[:, [1, 3]] *= img.shape[1]
            plotting.plot_boxes(scaled_pred_boxes, fig, edgecolor='y', linestyles='--')

            for pred_box, pred_class_id in zip(scaled_pred_boxes, pred_class_ids):
                plt.text(pred_box[0], pred_box[1], self.id_to_class[pred_class_id], c='y', bbox=dict(facecolor='k', alpha=0.4))

            if show_box_matches:
                box_matches, _, _, _ = MSODataset._match_boxes(boxes, pred_boxes)
                for i, match in enumerate(box_matches):
                    if match == -1:
                        continue
                    box_pair = np.stack([scaled_boxes[i], scaled_pred_boxes[match]])
                    max_points = np.max(box_pair, axis=0)
                    min_points = np.min(box_pair, axis=0)

                    if show_intersection:
                        box_point = (max_points[0], max_points[1])
                        width = (min_points[2] - box_point[0])
                        height = (min_points[3] - box_point[1])
                    else:
                        box_point = (min_points[0], min_points[1])
                        width = (max_points[2] - box_point[0])
                        height = (max_points[3] - box_point[1])
                    p = PatchCollection(
                        [Rectangle(box_point, width, height)],
                        facecolor='b', alpha=0.5, edgecolor=None)
                    ax = fig.get_axes()[0]
                    ax.add_collection(p)

        plt.title(title)

        plt.axis('off')
        plt.tight_layout()
        if save_figure:
            plt.savefig(os.path.join(self.plot_dir, f'sample_{index}.png'))
        if show_figure:
            plt.show()
        else:
            return fig, img

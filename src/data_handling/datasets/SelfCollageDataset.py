import logging
import math
import os
import random

import torch
import torch.nn.functional as F
from PIL import ImageDraw, Image
from matplotlib import pyplot as plt
from torchvision.transforms.functional import gaussian_blur

from .CountingDataset import CountingDataset
from .CustomImageNet import CustomImageNet
from .CustomSUN397 import CustomSUN397
from .NoiseDataset import NoiseDataset
from ..DatasetSplits import DatasetSplits
from ...util import transforms, Constants, plotting
from ...util.FeatureEncoder import FeatureEncoder
from ...util.box_util import get_box_from_binary_mask, box_xyxy_to_cxcywh
from ...util.misc import get_resize_and_cropping_transforms
from ...util.misc_enums import ConstructionModes, LabelDistributions, \
    BaseDatasets, Blending, DensityMapTypes, SyntheticShapes, SyntheticColours, ClusterSimilarity

LOGGER = logging.getLogger()


class SelfCollageDataset(CountingDataset):

    OBJ_BOX_FILE_PREFIX = 'obj_boxes'

    NUM_OBJ_CLASSES = -1

    def __init__(self, args, base_dir, weights_dir, device, normalise_transform=None, patch_size=16,
                 verbose=False, **kwargs):
        self.num_samples = args.num_samples
        self.cropping = args.cropping
        self.background_cropping = args.background_cropping
        self.cropping_ratio = args.cropping_ratio
        self.construction_mode = args.construction_mode
        self.patch_size = patch_size
        self.blending = args.blending
        self.max_shots = args.max_shots

        # used for density maps
        self.density_map_type = args.density_map_type
        self.density_scaling = args.density_scaling
        self.density_blur_factor = args.density_blur_factor

        # used for pasting/segmentation
        self.ignore_overlap = args.ignore_overlap

        # used for pasting
        self.min_paste_size = args.min_paste_size
        self.max_paste_size = args.max_paste_size
        self.reference_img_size = 224

        self.max_complete_tries = 20

        self.verbose = verbose
        self.img_net_path = args.img_net_path
        self.img_net_ign_split_subfolder = args.img_net_ign_split_subfolder
        self.img_net_val_gt_path = args.img_net_val_gt_path
        self.img_net_meta_file_path = args.img_net_meta_file_path
        self.use_img_net_clusters = args.use_img_net_clusters
        self.num_img_net_clusters = args.num_img_net_clusters
        self.img_net_filter_clusters = args.img_net_filter_clusters
        self.cluster_similarity = args.cluster_similarity
        self.sun_path = args.sun397_path
        self.sun_class_name_path = args.sun397_class_name_path
        self.use_horizontal_flip = args.use_horizontal_flip
        self.use_vertical_flip = args.use_vertical_flip
        self.colour_jitter_transform = None
        self.colour_jitter_prob = args.colour_jitter_prob
        if args.use_colour_jitter:
            colour_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            self.colour_jitter_transform = transforms.RandomApply(colour_jitter, p=self.colour_jitter_prob)
        self.object_wise_colour_jitter_transform = None
        if args.use_object_wise_colour_jitter:
            object_wise_colour_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
            self.object_wise_colour_jitter_transform = transforms.RandomApply(object_wise_colour_jitter, p=self.colour_jitter_prob)
        self.normalise_transform = normalise_transform
        self.base_datasets = args.base_datasets
        self.img_net_subset_size_factor = args.img_net_subset_size_factor

        if args.label_distr is LabelDistributions.at_least_3:
            self.label_distr = torch.ones(args.num_count_classes, dtype=torch.float)
            self.label_distr[:3] = 0
        else:
            raise ValueError(f'Unsupported label_distr={args.label_distr.name}')
        if len(self.label_distr) != args.num_count_classes:
            raise ValueError(f'Label distribution {args.label_distr.name} not supported for {args.num_count_classes} classes')

        self.min_num_obj_clusters = args.min_num_obj_clusters
        self.max_num_obj_clusters = args.max_num_obj_clusters
        self.cluster_weight = args.cluster_weight
        self.use_fully_random_cluster_sizes = args.use_fully_random_cluster_sizes
        self.use_constant_object_numbers = args.use_constant_object_numbers
        self.max_objs = args.num_count_classes - 1
        # ensure that there is always at least one object per cluster
        if self.max_num_obj_clusters > 1 and self.use_constant_object_numbers:
            self.label_distr[(-self.max_num_obj_clusters + 1):] = 0

        if self.label_distr[:self.min_num_obj_clusters+1].sum() > 0:
            raise ValueError(f'Label distribution {args.label_distr.name} does not support min_num_obj_clusters={self.min_num_obj_clusters}')
        if self.use_constant_object_numbers:
            if self.min_num_obj_clusters < 2:
                raise ValueError(f'When using constant object numbers, min_num_obj_clusters has to be at least 2, but is {self.min_num_obj_clusters}')
            if self.use_fully_random_cluster_sizes:
                raise ValueError(f'When using constant object numbers, use_fully_random_cluster_sizes has to be False')
        if args.min_shots == 0 and self.use_constant_object_numbers:
            if self.construction_mode is ConstructionModes.Synthetic_shapes:
                raise NotImplementedError('Zero shot construction not supported for synthetic shapes')
            self.zero_shot_probability = 1 / (args.max_shots + 1)
            LOGGER.info(f'min_shots is {args.min_shots} and use_constant_object_numbers is True, constructing zero-shot samples with a probability of {self.zero_shot_probability:.2f}')
            LOGGER.warning(f'Zero-shot samples might contain less than {args.max_num_obj_clusters} clusters')
            if not args.vary_shots_per_sample:
                raise ValueError('When using constant object numbers, vary_shots_per_sample has to be True to support zero-shot samples')
        else:
            self.zero_shot_probability = 0

        # set image size
        self.set_img_size(args.img_size)

        self.get_segmentations = False
        if self.construction_mode is ConstructionModes.Segmentations:
            self.get_segmentations = True

        if (self.density_map_type is DensityMapTypes.Segmentation or self.density_map_type is DensityMapTypes.BlurredSegmentation) and not self.get_segmentations:
            raise ValueError(f'DensityMapType {self.density_map_type.name} only supported when using segmentations')

        self.feature_encoder = FeatureEncoder(weights_dir, self.target_img_size, device, args.batch_size, args.num_workers, args.disable_tqdm)

        super().__init__(base_dir, args.num_count_classes, reference_crop_size=args.reference_crop_size,
                         use_reference_crops=True, **kwargs)

        self.copy_targets_last = args.copy_targets_last

        if self.label_distr[0] > 0 and not self.use_reference_crops:
            raise ValueError('use_reference_crops has to be True when training with samples with a count of 0')

        self.resize_after_segmentation = args.resize_after_segmentation
        if self.resize_after_segmentation and not self.get_segmentations:
            raise ValueError('resize_after_segmentation requires get_segmentations to be True')
        if self.resize_after_segmentation and not self.construction_mode is ConstructionModes.Segmentations:
            raise ValueError('resize_after_segmentation is only supported with construction_mode=Segmentations')
        self.correlated_object_sizes = args.correlated_object_sizes
        self.max_rel_size_deviation = args.max_rel_size_deviation
        self.absolute_min_paste_size = args.absolute_min_paste_size
        self.absolute_max_paste_size = args.absolute_max_paste_size

        # factor which determines the difficulty of the constructed samples
        # 0: easy, 1: hard
        self.difficulty_factor = 0

        if self.in_memory:
            LOGGER.warning(f'In_memory ignored for {self.__class__.__name__}')

        LOGGER.debug('PatchDataset setup complete')

    def __getitem__(self, index: int):
        num_obj = self.labels[index]
        if self.construction_mode is ConstructionModes.Synthetic_shapes:
            img, label_dict = self._synthesise_image(num_obj)
        else:
            img, label_dict = self._construct_image(num_obj)

        if self.transform is not None:
            img, target_dict = self.transform(img, label_dict)

        if self.colour_jitter_transform is not None:
            img = self.colour_jitter_transform(img)

        if self.normalise_transform is not None:
            img = self.normalise_transform(img)

        if self.use_reference_crops:
            # save rescaled image crops as references
            self._add_ref_imgs(img, label_dict, self.max_shots)

        return img, label_dict

    def _setup(self):

        def_img_net_path = os.path.join(self.base_dir, 'ImageNet')
        if self.img_net_path is None:
            self.img_net_path = def_img_net_path

        img_net_kwargs = {
            'root': self.img_net_path,
            'processed_meta_dir': def_img_net_path,
            'split': 'train',
            'subset_size_factor': self.img_net_subset_size_factor,
            'return_segmentation_masks': self.get_segmentations,
            'ign_split_subfolder': self.img_net_ign_split_subfolder,
            'img_net_val_gt_path': self.img_net_val_gt_path,
            'img_net_meta_file_path': self.img_net_meta_file_path,
            'use_clusters': self.use_img_net_clusters,
            'num_clusters': self.num_img_net_clusters,
            'feature_encoder': self.feature_encoder,
            'filter_clusters': self.img_net_filter_clusters
        }

        if self.construction_mode is ConstructionModes.Synthetic_shapes:
            LOGGER.info(f'No datasets are used when using construction mode {self.construction_mode.name}')
            self.obj_dataset = None
            self.zero_count_dataset = None
        else:
            if self.base_datasets is BaseDatasets.ImageNet_SUN:
                self.obj_dataset = CustomImageNet(**img_net_kwargs)
                self.zero_count_dataset = CustomSUN397(self.base_dir, split=DatasetSplits.TRAIN,
                                                       img_dir=self.sun_path, class_name_path=self.sun_class_name_path)
            elif self.base_datasets is BaseDatasets.ImageNet_StyleGAN_Noise:
                self.obj_dataset = CustomImageNet(**img_net_kwargs)
                self.zero_count_dataset = NoiseDataset(self.base_dir)
            elif self.base_datasets is BaseDatasets.ImageNet:
                self.obj_dataset = CustomImageNet(**img_net_kwargs)
                self.zero_count_dataset = self.obj_dataset
            else:
                raise ValueError(f'Object dataset "{self.base_datasets}" not supported')

    def set_img_size(self, img_size):
        self.target_img_size = img_size
        self.paste_size_scaling = self.target_img_size / self.reference_img_size

        trans, resize_img_size = get_resize_and_cropping_transforms(self.target_img_size, self.cropping,
                                                                    self.cropping_ratio)
        back_trans, back_resize_img_size = get_resize_and_cropping_transforms(self.target_img_size, self.background_cropping,
                                                                    self.cropping_ratio)
        if self.cropping:
            LOGGER.debug(f'Resizing images to {resize_img_size} before cropping')
        if self.background_cropping:
            LOGGER.debug(f'Resizing background images to {back_resize_img_size} before cropping')
        if self.use_horizontal_flip:
            trans += [transforms.RandomHorizontalFlip()]
            back_trans += [transforms.RandomHorizontalFlip()]
        if self.object_wise_colour_jitter_transform is not None:
            trans += [self.object_wise_colour_jitter_transform]
        if self.use_vertical_flip:
            raise NotImplementedError(f'RandomVerticalFlip is not implemented')
        self.patch_transform = transforms.Compose(trans)
        self.background_transform = transforms.Compose(back_trans)

    def set_difficulty_factor(self, epoch, num_epochs):
            """sets the difficulty factor for the current epoch"""
            if epoch < 0:
                # negative epochs are used for testing the model prior to any training and are therefore ignored
                return
            if num_epochs == 1:
                # if only one epoch is used, the difficulty factor is always 0
                self.difficulty_factor = 0
            else:
                self.difficulty_factor = -0.5 * math.cos(math.pi * epoch / (num_epochs-1)) + 0.5

    def _get_random_background_samples(self, num_samples, ign_clusters=None):
        if self.base_datasets is BaseDatasets.ImageNet and ign_clusters is not None:
            # get unique clusters that should be ignored
            unique_clusters = ign_clusters.unique().view(-1, 1)
            valid_idxs_mask = (self.zero_count_dataset.clusters.repeat(len(unique_clusters), 1) != unique_clusters).all(0)
            valid_idxs = torch.arange(len(self.zero_count_dataset.clusters))[valid_idxs_mask]
            if len(valid_idxs) < num_samples:
                LOGGER.warning(f'Not enough background samples to ignore clusters {ign_clusters}')
                valid_idxs = torch.arange(len(self.zero_count_dataset.clusters))
            background_idxs = valid_idxs[torch.randperm(len(valid_idxs))[:num_samples]]
        else:
            background_idxs = torch.randperm(len(self.zero_count_dataset))[:num_samples]
        background_samples = [self.background_transform(self.zero_count_dataset[idx][0]) for idx in background_idxs]
        return background_samples

    def _get_obj_img(self, index):
        return self.patch_transform(*self.obj_dataset[index])

    def _get_file_names(self):
        return [None for _ in range(self.num_samples)]

    def _get_labels(self):
        if hasattr(self, 'labels'):
            return self.labels
        else:
            return self._sample_label(self.num_samples)

    def _get_id_to_class(self):
        return getattr(self.obj_dataset, 'classes', None)

    def _get_img_path(self, index):
        return None

    def _sample_label(self, num_samples):
        return torch.multinomial(self.label_distr, num_samples, replacement=True)

    def _synthesise_image(self, num_objs):
        num_obj_per_type = self._get_num_per_cluster(num_objs)
        shape_sizes = self._get_correlated_sizes(num_obj_per_type.sum())
        type_idxs = torch.zeros((num_obj_per_type.sum(),), dtype=torch.long)
        cur_start = 0
        for i, num_obj in enumerate(num_obj_per_type):
            type_idxs[cur_start:cur_start+num_obj] = i
            cur_start += num_obj

        # get a random shape-colour combination for each cluster
        combinations = [(shape, colour) for shape in SyntheticShapes for colour in SyntheticColours]
        random.shuffle(combinations)
        combinations = combinations[:num_obj_per_type.shape[0]]

        unused_colours = [colour for colour in SyntheticColours if colour not in [c for _, c in combinations]]
        random.shuffle(unused_colours)
        background_colour = unused_colours[0]

        if self.copy_targets_last:
            type_idxs = type_idxs.flip(0)
            shape_sizes = shape_sizes.flip(0)

        all_obj_boxes = torch.zeros(len(shape_sizes), 4)

        # creating new Image object
        img = Image.new("RGB", (self.target_img_size, self.target_img_size), background_colour.name)
        draw = ImageDraw.Draw(img)
        for obj_idx, (type_idx, shape_size) in enumerate(zip(type_idxs, shape_sizes)):
            shape, colour = combinations[type_idx]
            size = int(shape_size.item() * self.paste_size_scaling)
            # get random position
            top_left_pos = torch.randint(self.target_img_size-size, (2,))
            center_pos = top_left_pos + size/2
            all_obj_boxes[obj_idx] = torch.cat([top_left_pos, top_left_pos+size])
            self._add_synthetic_object(draw, shape, colour.name, size, center_pos, outline_width=int(2*self.paste_size_scaling))

        if self.copy_targets_last:
            all_obj_boxes = all_obj_boxes.flip(0)
        obj_boxes = all_obj_boxes[:num_objs]

        # create density map
        density_map = self._create_density_map_from_boxes(obj_boxes)
        # create density map for all objects
        all_density_map = self._create_density_map_from_boxes(all_obj_boxes)
        # blur density maps
        density_map = self._blur_density_map(density_map, obj_boxes)
        all_density_map = self._blur_density_map(all_density_map, all_obj_boxes)

        target_dict = {self.LABEL_DICT_GLOBAL_COUNT: num_objs, self.LABEL_DICT_TOTAL_NUM_OBJECTS: len(shape_sizes)}
        obj_boxes = obj_boxes / self.target_img_size
        target_dict[self.LABEL_DICT_BOXES] = self.pad_sample(obj_boxes, max_num_elements=self.num_classes - 1,
                                                             elements_dim=4, dtype=torch.float32)
        all_obj_boxes = all_obj_boxes / self.target_img_size
        target_dict[self.LABEL_DICT_ALL_BOXES] = self.pad_sample(all_obj_boxes, max_num_elements=self.num_classes - 1,
                                                                 elements_dim=4, dtype=torch.float32)
        target_dict[self.LABEL_DICT_NUM_ELEMENTS] = obj_boxes.shape[0]
        target_dict[self.LABEL_DICT_CLASSES] = self.pad_sample(torch.full((len(shape_sizes),), -1), max_num_elements=self.num_classes - 1,
                                                               elements_dim=0, dtype=torch.int64)
        target_dict[self.LABEL_DICT_DENSITY_MAPS] = density_map
        target_dict[self.LABEL_DICT_ALL_DENSITY_MAPS] = all_density_map

        return transforms.ToTensor()(img), target_dict

    def _add_synthetic_object(self, draw, shape, colour, size, pos, outline_colour='black', outline_width=2):
        xy = [(pos[0]-size/2, pos[1]-size/2), (pos[0]+size/2, pos[1]+size/2)]
        if shape is SyntheticShapes.Square:
            draw.rectangle(xy, fill=colour, outline=outline_colour, width=outline_width)
        elif shape is SyntheticShapes.Circle:
            draw.ellipse(xy, fill=colour, outline=outline_colour, width=outline_width)
        elif shape is SyntheticShapes.Triangle:
            xy = [(pos[0]-size/2, pos[1]-size/2), (pos[0]+size/2, pos[1]-size/2), (pos[0], pos[1]+size/2)]
            draw.polygon(xy, fill=colour, outline=outline_colour, width=outline_width)
        else:
            raise NotImplementedError(f'Shape {shape} not implemented')

    def _get_num_per_cluster(self, num_objs, zero_shot_sample=False):
        # determine the number of object clusters, one of which is to be counted
        num_clusters = random.randint(min(num_objs, self.min_num_obj_clusters), min(num_objs, self.max_num_obj_clusters))

        if zero_shot_sample and not self.use_constant_object_numbers:
            raise NotImplementedError('Zero-shot sampling is only implemented for use_constant_object_numbers')

        if self.use_constant_object_numbers:

            num_obj_per_cluster = torch.ones((num_clusters,), dtype=torch.int64)
            if zero_shot_sample:
                # if use_constant_object_numbers and zero_shot_sample are True, the total number of objects is num_objs
                total_num_objs = num_objs
                # all clusters have a random number of objects
                rand_cls_idxs = torch.randperm(num_clusters)
            else:
                # if use_constant_object_numbers is True, the total number of objects is self.max_objs
                total_num_objs = self.max_objs
                # num_objs determines the number of objects in the target cluster
                num_obj_per_cluster[0] = num_objs
                # all non-target clusters have a random number of objects
                rand_cls_idxs = torch.randperm(num_clusters - 1) + 1
            num_remaining_objs = total_num_objs - num_obj_per_cluster.sum()
            for cls_idx in rand_cls_idxs[:-1]:
                num_obj_per_cluster[cls_idx] += random.randint(0, num_remaining_objs)
                num_remaining_objs = total_num_objs - num_obj_per_cluster.sum()
                if num_remaining_objs <= 0:
                    break
            num_obj_per_cluster[rand_cls_idxs[-1]] += num_remaining_objs
        elif self.use_fully_random_cluster_sizes:
            # get the number of objects per cluster
            num_obj_per_cluster = torch.ones((num_clusters,), dtype=torch.int64)
            rand_cls_idxs = torch.randperm(num_clusters)
            for cls_idx in rand_cls_idxs[:-1]:
                num_obj_per_cluster[cls_idx] += random.randint(0, num_objs - num_obj_per_cluster.sum())
            num_obj_per_cluster[rand_cls_idxs[-1]] += num_objs - num_obj_per_cluster.sum()

        else:
            # get weight per cluster
            cls_weights = [1] * num_clusters
            cls_weights[random.randint(0, len(cls_weights) - 1)] = self.cluster_weight
            # get the number of objects per cluster
            cls_idx, cls_count = torch.tensor(random.choices([i for i in range(num_clusters)], weights=cls_weights,
                                                             k=(num_objs - num_clusters))).unique(return_counts=True)
            num_obj_per_cluster = torch.ones((num_clusters,), dtype=torch.int64)
            num_obj_per_cluster[cls_idx] += cls_count
        return num_obj_per_cluster

    def _construct_image(self, num_objs):
        label = num_objs

        # get random object images
        if label == 0:
            raise NotImplementedError(f'construct_image is not implemented for {label=}')

        # construct a zero-shot image
        zero_shot_sample = self.zero_shot_probability > 0 and random.random() < self.zero_shot_probability

        num_obj_per_cluster = self._get_num_per_cluster(num_objs, zero_shot_sample=zero_shot_sample)

        # the total number of objects as opposed to the number of target objects
        # only relevant when using references
        total_num_objs = num_obj_per_cluster.sum()

        # get random clusters, making sure that there are enough objects in the clusters
        rand_clusters = torch.full((len(num_obj_per_cluster),), -1, dtype=torch.int64)
        for i, num_objs_in_cluster in enumerate(num_obj_per_cluster):

            # get clusters with enough objects
            _, filtered_mask = self.obj_dataset.get_filtered_clusters(num_objs_in_cluster, return_mask=True)

            if i > 0:
                # remove clusters that are already in rand_clusters
                filtered_mask[rand_clusters[:i]] = False

                if self.cluster_similarity is not ClusterSimilarity.No:
                    # filter clusters based on their similarity to target cluster
                    similar_mask = torch.zeros_like(filtered_mask)
                    if self.cluster_similarity.name.lower().startswith('top'):
                        top_x = int(self.cluster_similarity.name[3:])

                        similar_mask[self.obj_dataset.similar_cluster_ids[rand_clusters[0]][:top_x]] = True
                    elif self.cluster_similarity.name.lower().startswith('bottom'):
                        bottom_x = int(self.cluster_similarity.name[6:])
                        similar_mask[self.obj_dataset.similar_cluster_ids[rand_clusters[0]][-bottom_x:]] = True
                    elif self.cluster_similarity.name.lower().startswith('between'):
                        between = self.cluster_similarity.name[7:].split('_')
                        cluster_start = int(between[0])
                        cluster_end = int(between[1])
                        if len(between) == 3:
                            # change the lower bound based on the current difficulty factor
                            cluster_end = self.num_img_net_clusters - (self.num_img_net_clusters - cluster_end) * self.difficulty_factor
                            cluster_end = int(cluster_end)

                        similar_mask[self.obj_dataset.similar_cluster_ids[rand_clusters[0]][cluster_start:cluster_end]] = True
                    mask_candidate = filtered_mask & similar_mask
                    if mask_candidate.sum() < 1:
                        LOGGER.warning(f'No similar clusters found for target cluster {rand_clusters[0]} with at least {num_objs_in_cluster} objects')
                    else:
                        filtered_mask = mask_candidate

            filtered_clusters = self.obj_dataset.unique_clusters[filtered_mask]
            rand_clusters[i] = filtered_clusters[random.randint(0, len(filtered_clusters) - 1)]

        # get object images from clusters
        if not isinstance(self.obj_dataset.targets, torch.Tensor):
            target_tensor = torch.tensor(self.obj_dataset.targets)
        else:
            target_tensor = self.obj_dataset.targets
        obj_idxs = []
        if self.copy_targets_last:
            iterable = zip(rand_clusters.flip(0), num_obj_per_cluster.flip(0))
        else:
            iterable = zip(rand_clusters, num_obj_per_cluster)
        for rand_class, num_cls_objs in iterable:
            class_idxs = (target_tensor == rand_class).nonzero()
            obj_idxs.append(class_idxs[torch.randperm(len(class_idxs))[:num_cls_objs]])
        obj_idxs = torch.concat(obj_idxs, dim=0).view(-1)

        if not zero_shot_sample:
            label = num_obj_per_cluster[0]
            num_objs = num_obj_per_cluster[0]

        obj_samples = [self._get_obj_img(idx) for idx in obj_idxs]
        obj_imgs = [obj_sample[0] for obj_sample in obj_samples]
        obj_targets = [obj_sample[1] for obj_sample in obj_samples]
        obj_clusters = torch.tensor([obj_targets[i][Constants.TARGET_DICT_CLASSES] for i in range(total_num_objs)])

        density_map = None
        all_density_map = None

        if isinstance(obj_clusters, list):
            if len(obj_clusters) > 0:
                obj_clusters = torch.concat(obj_clusters)
            else:
                obj_clusters = torch.tensor(obj_clusters)

        if self.construction_mode is ConstructionModes.Pasting:
            background, constr_img, all_obj_boxes = self._construct_pasting_image(
                total_num_objs, obj_imgs, obj_clusters
            )
        elif self.get_segmentations:
            background, constr_img, all_obj_boxes, density_map, all_density_map = self._construct_segmentation_image(
                num_objs, total_num_objs, obj_imgs, obj_targets, obj_clusters
            )
        else:
            raise ValueError(f'Construction mode {self.construction_mode.name} not supported')

        if self.copy_targets_last:
            all_obj_boxes = all_obj_boxes.flip(0)
            obj_clusters = obj_clusters.flip(0)

        # only the objects of the target class are considered for the object boxes
        obj_boxes = all_obj_boxes[:num_objs]

        if self.density_map_type is DensityMapTypes.BoxCenters:
            # create density map
            density_map = self._create_density_map_from_boxes(obj_boxes)

            # create density map for all objects
            all_density_map = self._create_density_map_from_boxes(all_obj_boxes)

        if self.density_map_type is DensityMapTypes.BoxCenters or self.density_map_type is DensityMapTypes.BlurredSegmentation:
            # blur density maps
            density_map = self._blur_density_map(density_map, obj_boxes)
            all_density_map = self._blur_density_map(all_density_map, all_obj_boxes)

        target_dict = {self.LABEL_DICT_GLOBAL_COUNT: label, self.LABEL_DICT_TOTAL_NUM_OBJECTS: total_num_objs, self.LABEL_DICT_IS_ZERO_SHOT: zero_shot_sample}
        obj_boxes = obj_boxes / self.target_img_size
        target_dict[self.LABEL_DICT_BOXES] = self.pad_sample(obj_boxes, max_num_elements=self.num_classes - 1,
                                                             elements_dim=4, dtype=torch.float32)
        all_obj_boxes = all_obj_boxes / self.target_img_size
        target_dict[self.LABEL_DICT_ALL_BOXES] = self.pad_sample(all_obj_boxes, max_num_elements=self.num_classes - 1,
                                                             elements_dim=4, dtype=torch.float32)
        target_dict[self.LABEL_DICT_NUM_ELEMENTS] = obj_boxes.shape[0]
        target_dict[self.LABEL_DICT_CLASSES] = self.pad_sample(obj_clusters, max_num_elements=self.num_classes - 1,
                                                               elements_dim=0, dtype=torch.int64)
        if density_map is not None:
            target_dict[self.LABEL_DICT_DENSITY_MAPS] = density_map
            target_dict[self.LABEL_DICT_ALL_DENSITY_MAPS] = all_density_map

        return constr_img, target_dict

    def _create_density_map_from_boxes(self, boxes):
        # create density map
        obj_boxes_cxcywh = box_xyxy_to_cxcywh(boxes)
        density_map = torch.zeros((self.target_img_size, self.target_img_size), dtype=torch.float)
        density_map[obj_boxes_cxcywh[:, 1].long(), obj_boxes_cxcywh[:, 0].long()] = self.density_scaling
        return density_map

    def _construct_pasting_image(self, total_num_objs, obj_imgs, obj_clusters):

        imgs = obj_imgs.copy()

        # get random background images
        no_obj_imgs = self._get_random_background_samples(1, ign_clusters=obj_clusters)
        background = no_obj_imgs[0]
        if len(no_obj_imgs) > 1:
            imgs += no_obj_imgs[1:]

        background, constr_img, all_obj_boxes = self._paste_images(background, imgs, total_num_objs, len(obj_imgs))

        return background, constr_img, all_obj_boxes

    def _construct_segmentation_image(self, num_objs, total_num_objs, obj_imgs, obj_targets, obj_clusters):
        # get random background image
        background = self._get_random_background_samples(1, ign_clusters=obj_clusters)[0]

        imgs = obj_imgs.copy()

        constr_img, all_obj_boxes, density_map, all_density_map = self._paste_image_segmentations(background, imgs, obj_targets, num_objs, total_num_objs)

        return background, constr_img, all_obj_boxes, density_map, all_density_map

    def _paste_images(self, background: torch.Tensor, imgs: list[torch.Tensor], total_num_objs: int, num_obj_imgs: int):

        complete_try_num = 0
        max_img_try_num = 0
        constr_img = None
        obj_boxes = None
        for complete_try_num in range(self.max_complete_tries):

            # copy background
            constr_img = background.clone()

            # resize images
            if self.correlated_object_sizes:
                rand_patch_sizes = self._get_correlated_sizes(len(imgs))
            else:
                rand_patch_sizes = torch.randint(self.min_paste_size, self.max_paste_size, (len(imgs),))
            rand_patch_sizes_pixel = (rand_patch_sizes * self.paste_size_scaling).int()
            resized_imgs = [transforms.Resize((rand_patch_size, rand_patch_size))(img)
                            for img, rand_patch_size in zip(imgs, rand_patch_sizes_pixel)]
            # keep track of pasted regions
            paste_mask = torch.zeros(*constr_img.shape[1:3], dtype=torch.bool)

            # store boxes and clusters of pasted objects
            obj_boxes = torch.zeros(total_num_objs, 4)

            all_pasted = True
            max_img_try_num = 0

            box_points = torch.zeros(len(resized_imgs), 4)

            # paste each image
            for img_idx, (img, rand_img_size_target) in enumerate(zip(resized_imgs, rand_patch_sizes_pixel)):
                if self.copy_targets_last:
                    is_obj_img = (len(resized_imgs)-img_idx) <= num_obj_imgs
                else:
                    is_obj_img = img_idx < num_obj_imgs

                # create mask for possible pasting locations
                loc_paste_mask = torch.ones(*constr_img.shape[1:3], dtype=torch.bool)
                # exclude regions too close to the image border
                loc_paste_mask[-rand_img_size_target + 1:] = False
                loc_paste_mask[:, -rand_img_size_target + 1:] = False

                if img_idx > 0 and not self.ignore_overlap:
                    # exclude regions too close to other images
                    box_margin_coord = box_points.int()[:img_idx].clone()
                    box_margin_coord[:, :2] = torch.clip(box_margin_coord[:, :2] - rand_img_size_target + 1,
                                                         0)
                    for i in range(box_margin_coord.shape[0]):
                        loc_paste_mask[box_margin_coord[i, 1]:box_margin_coord[i, 3],
                        box_margin_coord[i, 0]:box_margin_coord[i, 2]] = False
                possible_locations = loc_paste_mask.nonzero()

                # test if there are free locations
                if possible_locations.shape[0] == 0:
                    # if this is the last try, copy it anywhere except for the border of the image
                    if complete_try_num == self.max_complete_tries - 1:
                        max_y_idx = paste_mask.shape[0] - rand_img_size_target + 1
                        max_x_idx = paste_mask.shape[1] - rand_img_size_target + 1
                        possible_locations = torch.stack([torch.arange(max_y_idx).repeat_interleave(max_x_idx),
                                                          torch.arange(max_x_idx).repeat(max_y_idx)]).T
                    else:
                        # start a new try
                        all_pasted = False
                        break
                y_start, x_start = possible_locations[random.randint(0, possible_locations.shape[0] - 1)]
                y_end = y_start + rand_img_size_target
                x_end = x_start + rand_img_size_target
                box_points[img_idx] = torch.tensor([x_start, y_start, x_end, y_end])
                if is_obj_img:
                    obj_boxes[img_idx] = torch.tensor([x_start, y_start, x_end, y_end])

                # paste image
                paste_mask[y_start:y_end, x_start:x_end] = True
                constr_img[:, y_start:y_end, x_start:x_end] = img

            if all_pasted:
                break

        if complete_try_num == self.max_complete_tries - 1:
            LOGGER.warning(
                f'Constructed sample with maximum number of tries, it might contain overlapping images')
        elif self.verbose:
            LOGGER.debug(
                f'Constructed sample with {complete_try_num} complete tries and a maximum of {max_img_try_num} tries per image')

        return background, constr_img, obj_boxes

    def _paste_image_segmentations(self, background: torch.Tensor, imgs: list[torch.Tensor], obj_targets,
                                   num_objs: int, total_num_objs: int):
        complete_try_num = 0
        max_img_try_num = 0

        constr_img = None
        all_obj_boxes = None
        density_map = None
        all_density_map = None

        for complete_try_num in range(self.max_complete_tries):

            # copy background
            constr_img = background.clone()

            # resize images
            if self.correlated_object_sizes:
                rand_patch_sizes = self._get_correlated_sizes(len(imgs))
            else:
                rand_patch_sizes = torch.randint(self.min_paste_size, self.max_paste_size, (len(imgs),))
            rand_patch_sizes_pixel = (rand_patch_sizes * self.paste_size_scaling).int()
            if not self.resize_after_segmentation:
                obj_samples = [transforms.Resize((rand_patch_size, rand_patch_size))(img, obj_target)
                               for img, obj_target, rand_patch_size in zip(imgs, obj_targets, rand_patch_sizes_pixel)]
            else:
                obj_samples = [(img, obj_target) for img, obj_target in zip(imgs, obj_targets)]

            # keep track of pasted regions
            paste_mask = torch.zeros(*constr_img.shape[-2:], dtype=torch.bool)

            if self.density_map_type is DensityMapTypes.Segmentation or self.density_map_type is DensityMapTypes.BlurredSegmentation:
                density_map = torch.zeros(*constr_img.shape[-2:], dtype=torch.float)
                all_density_map = torch.zeros(*constr_img.shape[-2:], dtype=torch.float)
            else:
                density_map = None
                all_density_map = None

            # store boxes and clusters of pasted objects
            all_obj_boxes = torch.zeros(total_num_objs, 4)

            all_pasted = True
            max_img_try_num = 0
            y_x_artifact_offset = None

            # paste each image
            for img_idx, obj_sample in enumerate(obj_samples):
                obj_img, obj_target = obj_sample
                # get the box of the object
                obj_box = get_box_from_binary_mask(obj_target['segmentations'])
                # remove the background from the image
                segmented_obj = obj_img * obj_target['segmentations']
                cut_obj = segmented_obj[:, obj_box[1]:obj_box[3], obj_box[0]:obj_box[2]]
                cut_seg_mask = obj_target['segmentations'][obj_box[1]:obj_box[3], obj_box[0]:obj_box[2]]

                if self.resize_after_segmentation:
                    # resize object, keeping the aspect ratio
                    rand_patch_size = rand_patch_sizes_pixel[img_idx]
                    try:
                        cut_obj, tmp_target_dict = transforms.Resize((rand_patch_size,), max_size=rand_patch_size+1)(cut_obj, {Constants.TARGET_DICT_SEGMENTATIONS: cut_seg_mask})
                    except RuntimeError as e:
                        max_dim = 0 if cut_obj.shape[1] > cut_obj.shape[2] else 1
                        new_resize_size = torch.ones(2, dtype=torch.int64)
                        new_resize_size[max_dim] = rand_patch_size
                        LOGGER.warning(f'Error resizing object with size {cut_obj.shape} to {rand_patch_size}, attempting to resize to {new_resize_size} instead. Original error: {e}')
                        cut_obj, tmp_target_dict = transforms.Resize(new_resize_size)(cut_obj, {Constants.TARGET_DICT_SEGMENTATIONS: cut_seg_mask})

                    cut_seg_mask = tmp_target_dict[Constants.TARGET_DICT_SEGMENTATIONS]

                is_obj = img_idx < total_num_objs
                if self.copy_targets_last:
                    is_target_obj = (len(obj_samples)-img_idx) <= num_objs
                else:
                    is_target_obj = img_idx < num_objs

                # get mask for valid pasting location, 0 = occupied, 1 = free
                if img_idx > 0 and not self.ignore_overlap:
                    # exclude regions too close to other objects or the border
                    conv_input = paste_mask.float().unsqueeze(0)
                    kernel = cut_seg_mask.unsqueeze(0).unsqueeze(0).float()
                    close_objs_mask = F.conv2d(conv_input, kernel, padding=0)
                    # binarize mask
                    close_objs_mask = close_objs_mask.squeeze(0) == 0
                    # create final mask, areas not covered by the convolution result are too close to the image border
                    loc_paste_mask = torch.zeros(paste_mask.shape, dtype=torch.bool)
                    loc_paste_mask[:close_objs_mask.shape[0], :close_objs_mask.shape[1]] = close_objs_mask
                else:
                    loc_paste_mask = torch.ones(*constr_img.shape[1:3], dtype=torch.bool)
                    # exclude regions too close to the image border
                    loc_paste_mask[-cut_obj.shape[1] + 1:] = False
                    loc_paste_mask[:, -cut_obj.shape[2] + 1:] = False

                possible_locations = loc_paste_mask.nonzero()

                # test if there are free locations
                if possible_locations.shape[0] == 0:
                    # if this is the last try, copy it anywhere except for the border of the image
                    if complete_try_num == self.max_complete_tries - 1:
                        max_y_idx = paste_mask.shape[0] - cut_obj.shape[1] + 1
                        max_x_idx = paste_mask.shape[1] - cut_obj.shape[2] + 1
                        possible_locations = torch.stack(
                            [torch.arange(max_y_idx).repeat_interleave(max_x_idx),
                             torch.arange(max_x_idx).repeat(max_y_idx)]).T
                    else:
                        # start a new try
                        all_pasted = False
                        break
                # get random location
                y_start, x_start = possible_locations[random.randint(0, possible_locations.shape[0] - 1)]
                y_end = y_start + cut_obj.shape[1]
                x_end = x_start + cut_obj.shape[2]

                if is_obj:
                    # paste object
                    if self.blending is Blending.No:
                        constr_img[:, y_start:y_end, x_start:x_end][cut_seg_mask.repeat((3, 1, 1))] = cut_obj[
                            cut_seg_mask.repeat((3, 1, 1))]
                    else:
                        raise NotImplementedError(f'Blending {self.blending} not implemented')
                else:
                    # create background artifact
                    background_artifact = background[:, y_start + y_x_artifact_offset[0]:y_end + y_x_artifact_offset[0],
                                          x_start + y_x_artifact_offset[1]:x_end + y_x_artifact_offset[1]]
                    # paste artifact
                    constr_img[:, y_start:y_end, x_start:x_end][cut_seg_mask.repeat((3, 1, 1))] = background_artifact[
                        cut_seg_mask.repeat((3, 1, 1))].clone()
                paste_mask[y_start:y_end, x_start:x_end] = paste_mask[y_start:y_end, x_start:x_end] | cut_seg_mask

                if density_map is not None and is_obj:
                    # if this is an object, add it to the density map for all objects
                    cut_density_map = cut_seg_mask.float()
                    density_normalisation = cut_density_map.sum() / self.density_scaling
                    cut_density_map = cut_density_map / density_normalisation
                    all_density_map[y_start:y_end, x_start:x_end] = all_density_map[y_start:y_end,
                                                                x_start:x_end] + cut_density_map
                    if is_target_obj:
                        # if this is a target object, add it to the density map for target objects
                        density_map[y_start:y_end, x_start:x_end] = density_map[y_start:y_end,
                                                                    x_start:x_end] + cut_density_map

                if is_obj:
                    all_obj_boxes[img_idx] = torch.tensor([x_start, y_start, x_end, y_end])

            if all_pasted:
                break

        if complete_try_num == self.max_complete_tries - 1:
            LOGGER.warning(
                f'Constructed sample with maximum number of tries, it might contain overlapping images')
        elif self.verbose:
            LOGGER.debug(
                f'Constructed sample with {complete_try_num} complete tries and a maximum of {max_img_try_num} tries per image')
        return constr_img, all_obj_boxes, density_map, all_density_map

    def _get_correlated_sizes(self, num_sizes):
        mean_object_size = torch.randint(self.min_paste_size, self.max_paste_size, (1,)).item()
        rand_patch_sizes = torch.full((num_sizes,), mean_object_size)
        size_variance = int(self.max_rel_size_deviation * mean_object_size)
        if size_variance == 0:
            size_variance = 1
        rand_patch_sizes = rand_patch_sizes + torch.randint(-size_variance, size_variance, (num_sizes,))
        rand_patch_sizes = torch.clamp(rand_patch_sizes, self.absolute_min_paste_size, self.absolute_max_paste_size)
        return rand_patch_sizes

    def _blur_density_map(self, density_map, obj_boxes=None):
        # blur density map
        if obj_boxes is None:
            # predefined kernel size and sigma
            sigma = 1 * self.density_blur_factor
            kernel_size = self._get_kernel_size_for_sigma(sigma)
            sigma = [sigma, sigma]
        else:
            # kernel size and sigma based on box sizes
            obj_boxes_cxcywh = box_xyxy_to_cxcywh(obj_boxes)
            kernel_size = obj_boxes_cxcywh[:, -2:].mean(0).long()
            # modify kernel based on density blur factor, this corresponds to multiplying sigma by the blur factor
            #kernel_size = kernel_size * self.density_blur_factor
            odd_offset = (~(kernel_size % 2).to(torch.bool)).long()
            kernel_size = kernel_size + odd_offset
            sigma = (kernel_size - 1) / 8
            sigma = sigma * self.density_blur_factor
            kernel_size = self._get_kernel_size_for_sigma(sigma)
            kernel_size = list(kernel_size.numpy())
            sigma = list(sigma.numpy())

        density_map = gaussian_blur(density_map.unsqueeze(0), kernel_size, sigma).squeeze(0)
        return density_map

    def _get_kernel_size_for_sigma(self, sigma):
        if isinstance(sigma, torch.Tensor):
            if sigma.dim() == 0:
                sigma = sigma.item()
            else:
                return torch.tensor([self._get_kernel_size_for_sigma(s.item()) for s in sigma])
        return 2 * round(4 * sigma) + 1

    def _load_image(self, index):
        raise NotImplementedError(f'_load_image is not implemented for {self.__class__.__name__}, use __getitem__ or _construct_image instead.')

    def visualise_sample(self, index, show_boxes=True, show_density_map=False, save_figure=True,
                         show_references=False, show_figure=True, show_all_objects=False, show_title=True,
                         max_num_boxes=None, show_clusters=False):
        if not isinstance(index, list):
            index = [index]

        if show_all_objects and not show_density_map:
            LOGGER.warning(f'detailed is not possible without {show_density_map=} ')
            show_all_objects = False

        imgs = []
        target_dicts = []
        for idx in index:
            img, target_dict = self[idx]
            imgs.append(img)
            target_dicts.append(target_dict)

        for i, (idx, img, target_dict) in enumerate(zip(index, imgs, target_dicts)):
            label = target_dict[self.LABEL_DICT_GLOBAL_COUNT]
            if show_all_objects:
                fig = plt.figure(figsize=(12, 6))
            else:
                fig = plt.figure(figsize=(6, 6))
            title = f'Label: {label}'
            if len(img.shape) == 4:
                num_patches = self.target_img_size // self.patch_size
                img = img.transpose(0, 1).view(img.shape[1], num_patches, num_patches, self.patch_size, self.patch_size)
                img = img.transpose(2, 3).reshape(-1, self.target_img_size, self.target_img_size)

            if show_all_objects:
                plt.subplot(1, 2, 1)
            plt.imshow(img.permute((1, 2, 0)))
            if show_title:
                plt.title(title)

            if show_boxes:
                boxes = CountingDataset.unpad_sample(target_dict[self.LABEL_DICT_BOXES])
                if max_num_boxes is not None:
                    boxes = boxes[:max_num_boxes]
                self._add_box_visualisation(fig, img, boxes, target_dict, show_clusters=show_clusters, linewidths=2)

            if show_density_map and self.LABEL_DICT_DENSITY_MAPS in target_dict:
                all_density_map = target_dict[self.LABEL_DICT_ALL_DENSITY_MAPS]
                density_map = target_dict[self.LABEL_DICT_DENSITY_MAPS]

                if show_all_objects:
                    max_val = max(density_map.max(), all_density_map.max())
                else:
                    max_val = density_map.max()
                plt.imshow(density_map, alpha=0.5, cmap='jet', vmax=max_val)

                if show_all_objects:
                    plt.axis('off')
                    plt.tight_layout()
                    ax = plt.subplot(1, 2, 2)
                    plt.imshow(img.permute((1, 2, 0)))
                    all_density_map = target_dict[self.LABEL_DICT_ALL_DENSITY_MAPS]
                    plt.imshow(all_density_map, alpha=0.5, cmap='jet', vmax=max_val)
                    if show_boxes:
                        all_boxes = CountingDataset.unpad_sample(target_dict[self.LABEL_DICT_ALL_BOXES])
                        self._add_box_visualisation(fig, img, all_boxes, target_dict, show_clusters=show_clusters, ax=ax)
                    if show_title:
                        plt.title(f'All objects: {target_dict[self.LABEL_DICT_TOTAL_NUM_OBJECTS]}')

            plt.axis('off')
            plt.tight_layout()
            if save_figure:
                plt.savefig(os.path.join(self.plot_dir, f'sample_{idx}_{self.construction_mode.name}.png'))
            if show_figure:
                plt.show()

            if show_references:
                ref_images = self.unpad_sample(target_dict[self.LABEL_DICT_REF_IMGS])
                fig = plt.figure(figsize=(3*ref_images.shape[0], 3))
                for j, ref_img in enumerate(ref_images):
                    plt.subplot(1, ref_images.shape[0], j+1)
                    plt.imshow(ref_img.permute((1, 2, 0)))
                    plt.axis('off')
                    if show_title:
                        plt.title(f'Reference {j}')
                plt.tight_layout()
                if save_figure:
                    plt.savefig(os.path.join(self.plot_dir, f'{idx}_ref_imgs.png'))
                if show_figure:
                    plt.show()

    def _add_box_visualisation(self, fig, img, boxes, target_dict, show_clusters=True, **kwargs):
        boxes = boxes.clone()
        boxes[:, [0, 2]] *= img.shape[-1]
        boxes[:, [1, 3]] *= img.shape[-2]
        if len(boxes) > 0:
            # plot annotated boxes
            plotting.plot_boxes(boxes, fig, **kwargs)
            if show_clusters:
                class_labels = CountingDataset.unpad_sample(target_dict[self.LABEL_DICT_CLASSES])
                for pred_box, pred_class_id in zip(boxes, class_labels):
                    cls_name = self.id_to_class[pred_class_id]
                    if isinstance(cls_name, tuple):
                        cls_name = cls_name[0]
                    plt.text(pred_box[0], pred_box[1], cls_name, c='y',
                             bbox=dict(facecolor='k', alpha=0.4))


import argparse
import copy
import datetime
import json
import logging
import os
import random
from enum import Enum

from .misc import match_str_and_enum
from .misc_enums import ConstructionModes, LabelDistributions, BaseDatasets, Blending, DensityMapTypes, ClusterFilter, \
    ClusterSimilarity
from .optimisation import SupportedScheduler
from ..data_handling.SupportedDatasets import SupportedEvalDatasets
from ..models.ModelEnums import Heads, PretrainedWeights, ModelTypes, BackboneTypes
from ..models.backbones import BackboneFreezing

LOGGER = logging.getLogger()


def parse_args(return_defaults=False):
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser()

    # paths and file names
    parser.add_argument('--data_dir', default='', type=str, help='Path to the root data directory.')
    parser.add_argument('--img_net_path', default='', type=str, help='Path to the ImageNet directory.')
    parser.add_argument('--sun397_path', default='', type=str, help='Path to the directory containing the SUN397 image directories.')
    parser.add_argument('--sun397_class_name_path', default='', type=str, help='Path to the ClassName.txt file of SUN397')
    parser.add_argument('--img_net_ign_split_subfolder', action='store_true', help='Ignore split subfolder. If True, img_net_path has to point directly to the folder containing the images of the relevant split.')
    parser.add_argument('--img_net_val_gt_path',  default='', type=str, help='Path to the validation ground truth txt file for imagenet')
    parser.add_argument('--img_net_meta_file_path',  default='', type=str, help='Path to the meta file for imagenet')
    parser.add_argument('--model_folder_prefix', default='', type=str, help='Prefix added to model folders')
    parser.add_argument('--model_folder_postfix', default='', type=str, help='Postfix added to model folders')

    parser.add_argument('--pytorch_amp', action='store_true', help='Use automatic mixed precision training')
    parser.add_argument('--debug', action='store_true', help='Activate debugging enabling more detailed tensorboard logging')

    # setup
    parser.add_argument('--density_scaling', default=3000., type=float, help='Scaling used for the density map. The total number of objects corresponding to a density map is derived by taking the sum of the map divided by this scaling.')
    parser.add_argument('--num_count_classes', default=21, type=int, help='Number of count classes (includes 0)')

    # ImageNet
    parser.add_argument('--dont_use_img_net_clusters', dest='use_img_net_clusters', action='store_false', help='Use ImageNet classes instead of clusters.')
    parser.add_argument('--num_img_net_clusters', default=10000, type=int, help='Number of clusters used for ImageNet')
    parser.add_argument('--img_net_filter_clusters', default='no', type=str, help='Which filter to apply to the clusters, can only be used if use_img_net_clusters is True.')

    # PatchDataset
    parser.add_argument('--img_size', default=224, type=int, help='Image size used for resizing/cropping in PatchDataset')
    parser.add_argument('--cropping', action='store_true', help='Use cropping in PatchDataset')
    parser.add_argument('--background_cropping', action='store_true', help='Use cropping for backgrounds in PatchDataset')
    parser.add_argument('--cropping_ratio', default=0.875, type=float, help='Ratio (cropped image size)/(resize size). When using cropping, images are first rescaled to img_size/cropping_ratio.')
    parser.add_argument('--no_horizontal_flip', dest='use_horizontal_flip', action='store_false', help='Don\'t use horizontal flip in PatchDataset')
    parser.add_argument('--use_vertical_flip', action='store_true', help='Use vertical flip in PatchDataset')
    parser.add_argument('--use_colour_jitter', action='store_true', help='Use colour jitter for constructed samples in PatchDataset')
    parser.add_argument('--use_object_wise_colour_jitter', action='store_true', help='Use colour jitter on independently on each object in PatchDataset')
    parser.add_argument('--colour_jitter_prob', default=0.0, type=float, help='Probability of appying colour jitter, ignored if use_colour_jitter is False')
    parser.add_argument('--dont_normalise', dest='normalise', action='store_false', help='Don\'t normalise images in PatchDataset')
    parser.add_argument('--dont_ignore_overlap', dest='ignore_overlap', action='store_false', help='Dont ignore overlap with other objects when constructing images using pasting or segmentation')
    parser.add_argument('--label_distr', default='at_least_3', type=str, help='Distribution of counts in the PatchDataset')
    parser.add_argument('--construction_mode', default='segmentations', type=str, help='Construction mode of training images')
    parser.add_argument('--base_datasets', default='ImageNet_SUN', type=str, help='Name of the base datasets used in PatchDataset')
    parser.add_argument('--blending', default='no', type=str, help='Type of blending used for constructing images. Only used if construction_mode is based on segmentations.')
    parser.add_argument('--density_map_type', default='boxcenters', type=str, help='Type of the density map ground truths.')
    parser.add_argument('--density_blur_factor', default=1, type=float, help='How much to blur the density map ground truths. 1 is the default blur')
    parser.add_argument('--dont_resize_after_segmentation', dest='resize_after_segmentation', action='store_false', help='Apply the min/max size resizing to the full image instead of the segmented object.')
    # references
    parser.add_argument('--dont_copy_targets_last', dest='copy_targets_last', action='store_false', help='Copy non-target objects after target objects.')
    parser.add_argument('--min_num_obj_clusters', default=2, type=int, help='Minimum number of object clusters when using references (if an image contains fewer than min_num_obj_clusters objects, the total number of objects is used instead)')
    parser.add_argument('--max_num_obj_clusters', default=2, type=int, help='Maximum number of object clusters when using references')
    parser.add_argument('--cluster_weight', default=1, type=float, help='Weight of one of the clusters when computing the number of objects per cluster. A weight of 1 corresponds to equal weights while values higher than 1 lead to a skewed distribution.')
    parser.add_argument('--use_fully_random_cluster_sizes', action='store_true', help='Use fully random cluster sizes')
    parser.add_argument('--cluster_similarity', default='no', type=str, help='Specifies how cluster similarity is used when selecting non-target clusters')
    parser.add_argument('--dont_use_constant_object_numbers', dest='use_constant_object_numbers', action='store_false', help='Dont paste the maximum number of objects for every image, only changing the number in the target cluster')
    parser.add_argument('--reference_crop_size', default=64, type=int, help='Size of the reference images')
    parser.add_argument('--min_shots', default=1, type=int, help='Minimum number of shots')
    parser.add_argument('--max_shots', default=3, type=int, help='Maximum number of shots')
    # pasting
    parser.add_argument('--min_paste_size', default=15, type=int, help='Minimum size (in pixels) of pasted images, refers to an 224x224 image and will be scaled accordingly')
    parser.add_argument('--max_paste_size', default=70, type=int, help='Maximum size (in pixels) of pasted images, refers to an 224x224 image and will be scaled accordingly')
    parser.add_argument('--uncorrelated_object_sizes', dest='correlated_object_sizes', action='store_false', help='Don\'t use correlated object sizes which selects a mean objects size for each image based on min/max_paste_size.')
    parser.add_argument('--max_rel_size_deviation', default=0.3, type=float, help='Maximum relative deviation from the mean size. Only used if correlated_object_sizes is True')
    parser.add_argument('--absolute_min_paste_size', default=5, type=int, help='Absolute minimum size (in pixels) that every object should at least have when using correlated object sizes, refers to an 224x224 image and will be scaled accordingly')
    parser.add_argument('--absolute_max_paste_size', default=220, type=int, help='Absolute maximum size (in pixels) that every object should at most have when using correlated object sizes, refers to an 224x224 image and will be scaled accordingly')

    # model
    parser.add_argument('--model_type', default='UnCo', type=str, help='Type of model used')
    parser.add_argument('--head', default='linear', type=str, help='Type of head used by the model')
    parser.add_argument('--weights', default='dino', type=str, help='Type of weights used for the backbone')
    # regression
    parser.add_argument('--regression_act_fn', default='relu', type=str, help='Final activation function used for regression heads')
    # backbone
    parser.add_argument('--backbone_type', default='vit_b_16', type=str, help='Type of backbone used by the model')
    parser.add_argument('--backbone_freezing', default='completely', type=str, help='Type specifying how much of the backbone should be frozen')
    parser.add_argument('--num_frozen_blocks', default=10, type=int, help='Number of frozen transformer blocks, only used if backbone_freezing is block.')
    # few-shot model
    parser.add_argument('--dont_share_encoder', dest='share_encoder', action='store_false', help='Dont use the same encoder for the exemplar and the query images')
    parser.add_argument('--use_exemplar_attention', action='store_true', help='Uses gated cross-attention to modify the exemplar features')
    parser.add_argument('--use_exemplar_roi_align', action='store_true', help='Uses RoI Align to extract the exemplar features, only used if share_encoder is False')
    parser.add_argument('--use_exemplar_cls_token', action='store_true', help='Uses the CLS token of the exemplar, only used when share_encoder is True')
    parser.add_argument('--split_map_and_count', action='store_true', help='Predict the normalised density and count first instead of predicting the density map directly')
    parser.add_argument('--weigh_by_similarity', action='store_true', help='Weight encoded query patches by their maximum similarity to the exemplar')
    parser.add_argument('--use_similarity_projection', action='store_true', help='Add projected elementwise maximum similarity between query patches and exemplars')
    parser.add_argument('--unified_fim', action='store_true', help='Process exemplars and query images together using self-attention')
    parser.add_argument('--return_backbone_layer_features', default=-1, type=int, help='Use the features of this layer of the backbone in addition to the final features. If -1, only the final features are used.')

    # training
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--num_samples', default=10000, type=int, help='Number of samples constructed per epoch')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--dont_drop_last', dest='drop_last', action='store_false', help='Don\'t use drop_last for training dataloader.')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')
    parser.add_argument('--backbone_lr', default=5e-4, type=float, help='Learning rate used for the backbone (if not froozen)')
    parser.add_argument('--min_lr', default=0.0, type=float, help='Learning rate')
    parser.add_argument('--scale_lr', action='store_true', help='Linearly scale learning rate compared to a batch size of 128')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='Weight decay')
    parser.add_argument('--scheduler', default='Cosine_Scheduler', type=str, help='Type of scheduler used for training')
    parser.add_argument('--warmup_percentage', default=0.1, type=float, help='Percentage of epochs used for warmup')
    parser.add_argument('--vary_shots_per_sample', action='store_true', help='Use different number of shots for each sample instead of each batch')
    # multi-stage training
    parser.add_argument('--use_second_stage', action='store_true', help='Use second training stage')
    parser.add_argument('--second_img_size', default=384, type=int, help='Image size used for resizing/cropping in PatchDataset for the second training stage')
    parser.add_argument('--second_stage_percentage', default=0.1, type=float, help='Percentage of the total epochs used for the second stage of training')
    parser.add_argument('--reset_optimisation', action='store_true', help='Dont reset optimiser and scheduler when entering the second stage.')

    # density loss
    parser.add_argument('--density_loss_mask_prob', default=0.2, type=float, help='Probability of masking out pixels')
    parser.add_argument('--density_loss_dont_use_independent_masks', dest='density_loss_use_independent_masks', action='store_false', help='Draw a single mask for the whole batch instead of a random mask for each sample in a batch.')
    parser.add_argument('--density_loss_dont_keep_object_pixels', dest='density_loss_keep_object_pixels', action='store_false', help='Mask out pixels without checking if they are part of the target objects')
    parser.add_argument('--density_loss_keep_all_object_pixels', action='store_true', help='Only mask out pixels that are not part of any object')
    parser.add_argument('--density_loss_penalise_wrong_cluster_objects', action='store_true', help='Penalise including objects of other clusters in the density map')
    parser.add_argument('--density_loss_wrong_cluster_penality', default=2, type=float, help='Probability of masking out pixels')

    # miscellaneous
    parser.add_argument('--no_eval_before_training', dest='eval_before_training', action='store_false', help='Evaluate the model once before training.')
    parser.add_argument('--log_classification_metrics', action='store_true', help='Log classification metrics.')
    parser.add_argument('--ign_existing', action='store_true', help='Ignores existing models and trains a new one')
    parser.add_argument('--eval_dataset', default='evaldefault', type=str, help='Name of the evaluation dataset')
    parser.add_argument('--img_net_subset_size_factor', default=1, type=float, help='Factor used to determine the size of the ImageNet subset used. Ignored if >= 1')
    parser.add_argument('--seed', default=9832, type=int, help='Seed for pseudorandom numbers')
    parser.add_argument('--set_worker_seeds', action='store_true', help='Set different seeds for each dataloader worker during training')
    parser.add_argument('--dont_visualise_test', dest='visualise_test', action='store_false', help='Visualise predictions on the test set during training')
    parser.add_argument('--disable_tqdm', action='store_true', help='Disable TQDM')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers')

    if return_defaults:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    # convert parameters to enum elements
    args.eval_dataset = match_str_and_enum(args.eval_dataset, SupportedEvalDatasets)
    args.img_net_filter_clusters = match_str_and_enum(args.img_net_filter_clusters, ClusterFilter)
    args.cluster_similarity = match_str_and_enum(args.cluster_similarity, ClusterSimilarity)
    args.label_distr = match_str_and_enum(args.label_distr, LabelDistributions)
    args.construction_mode = match_str_and_enum(args.construction_mode, ConstructionModes)
    args.base_datasets = match_str_and_enum(args.base_datasets, BaseDatasets)
    args.blending = match_str_and_enum(args.blending, Blending)
    args.density_map_type = match_str_and_enum(args.density_map_type, DensityMapTypes)
    args.model_type = match_str_and_enum(args.model_type, ModelTypes)
    args.backbone_type = match_str_and_enum(args.backbone_type, BackboneTypes)
    args.backbone_freezing = match_str_and_enum(args.backbone_freezing, BackboneFreezing)
    args.head = match_str_and_enum(args.head, Heads)
    args.weights = match_str_and_enum(args.weights, PretrainedWeights)
    args.scheduler = match_str_and_enum(args.scheduler, SupportedScheduler)

    LOGGER.debug('Parsed arguments')
    return args


def get_model_dict(args):
    model_dict = {}
    ign_keys = ['data_dir', 'img_net_path', 'model_folder_prefix', 'model_folder_postfix', 'ign_existing',
                'eval_dataset', 'visualise', 'disable_tqdm', 'num_workers', 'debug', 'visualise_test',
                'log_classification_metrics']
    for key, value in args.__dict__.items():
        if key not in ign_keys:
            if isinstance(value, Enum):
                value = value.name
            model_dict[key] = value
    return model_dict


def fill_missing_values_with_defaults(args):
    def_args = parse_args(return_defaults=True)
    args = copy.deepcopy(args)
    missing_keys = []
    for k, v in def_args.__dict__.items():
        if k not in args:
            missing_keys.append(k)
            setattr(args, k, v)
    LOGGER.debug(f'Filled {len(missing_keys)} missing arguments with defaults: {missing_keys}')
    return args


def find_existing_model(models_dir, model_dict, model_dict_file, use_defaults=True):
    default_dict = get_model_dict(parse_args(return_defaults=True))
    for el in os.listdir(models_dir):
        hparams_file_path = os.path.join(models_dir, el, model_dict_file)
        if os.path.isfile(hparams_file_path):
            with open(hparams_file_path) as f:
                stored_hparams = json.load(f)
            if use_defaults:
                tmp = copy.deepcopy(default_dict)
                tmp.update(stored_hparams)
                stored_hparams = tmp

            if stored_hparams == model_dict:
                model_path = os.path.join(models_dir, el)
                return model_path
    return None


def get_model_dir(args, models_dir, model_dict, model_dict_file):
    # determine (existing or new) model directory
    if not args.ign_existing:
        stored_model_dir = find_existing_model(models_dir, model_dict, model_dict_file)
    else:
        stored_model_dir = None
    if stored_model_dir is None:
        model_name = args.model_type.name
        # Create a new model directory that is not already used
        if args.model_folder_prefix != '':
            model_name = args.model_folder_prefix + '_' + model_name
        if args.model_folder_postfix != '':
            model_name += '_' + args.model_folder_postfix
        model_name += '_'+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model_dir = os.path.join(models_dir, model_name)
        i = 1
        while os.path.isdir(model_dir):
            model_dir = os.path.join(models_dir, model_name + f'_{i}_{random.randint(0, 1e6)}')
            i += 1
        LOGGER.info(f'Creating model dir: {model_dir}')
        os.mkdir(model_dir)

        # save hyperparameters
        with open(os.path.join(model_dir, model_dict_file), 'w') as f:
            json.dump(model_dict, f, indent=4)
    else:
        model_dir = stored_model_dir
        LOGGER.info(f'Using saved model dir: {model_dir}')

    return model_dir

import logging
import math
import os
import pathlib
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.join(cur_dir, 'src', 'third_party', 'selfmask', 'networks')))

from src.data_handling.datasets.CustomImageNet import CustomImageNet
from src.util import transforms
from src.util.misc import get_resize_and_cropping_transforms
from src.third_party.selfmask.utils.misc import get_model, set_seeds
from src.third_party.selfmask.base_structure import BaseStructure

"""
Based on https://github.com/NoelShin/selfmask/blob/master/evaluator.py
"""

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
LOGGER = logging.getLogger()


class Annotator(BaseStructure):
    SEGMENTATIONS_FILE_PREFIX = 'segmentations'

    def __init__(
            self,
            network: callable,
            device,
            arch: str = "vit_small",
            debug: bool = False,
            denormalise: Optional[callable] = None
    ):
        super(Annotator, self).__init__(model=network, device=device)
        self.arch: str = arch
        self.debug: bool = debug
        self.network: callable = network
        self.denormalise: Optional[callable] = denormalise

    def visualise(self, idx, img, img_cls, pred_mask, dir_ckpt):
        from matplotlib import pyplot as plt

        plot_img = self.denormalise(img) if self.denormalise is not None else img
        plot_img = plot_img.permute(1, 2, 0)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(plot_img)
        plt.axis('off')
        plt.tight_layout()
        plt.subplot(1, 2, 2)
        plt.imshow(plot_img*pred_mask[..., None])
        plt.axis('off')
        plt.suptitle(img_cls)
        plt.tight_layout()
        plt.savefig(os.path.join(dir_ckpt, f'segmentation_{idx}.png'))
        plt.show()

    @torch.no_grad()
    def __call__(
            self,
            dataset,
            seg_dir: str,
            plot_dir: str,
            img_size: Optional[int] = None,
            scale_factor: int = 2,
            batch_size: int = 1,
            num_workers: int = 0,
            max_num_batches_per_file: int = 1,
            disable_tqdm: bool = False
    ):
        use_multiple_files = max_num_batches_per_file is not None

        segmentations_file_postfix = '.pt'
        segmentations_file_prefix = self.SEGMENTATIONS_FILE_PREFIX + f'_{img_size}'

        img_data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        LOGGER.debug(f'Creating segmentations for {len(dataset)} images')

        if use_multiple_files:
            files = [f for f in os.listdir(seg_dir) if f.startswith(segmentations_file_prefix) and f.endswith(segmentations_file_postfix)]
            total_num_files = math.ceil(len(img_data_loader) / max_num_batches_per_file)
            if len(files) == total_num_files:
                LOGGER.info(f'Using segmentations saved in {seg_dir}')
                return
        else:
            segmentations_file_name = segmentations_file_prefix+segmentations_file_postfix

            segmentations_file_path = os.path.join(seg_dir, segmentations_file_name)
            if os.path.isfile(segmentations_file_path):
                LOGGER.info(f'Using segmentations saved in {segmentations_file_path}')
                return

        if use_multiple_files:
            obj_segments = torch.zeros((max_num_batches_per_file * batch_size, img_size, img_size), dtype=torch.bool)
        else:
            obj_segments = torch.zeros((len(dataset), img_size, img_size), dtype=torch.bool)
        num_batches_in_file = 0

        for batch_idx, (imgs, target) in enumerate(tqdm(img_data_loader, leave=False, desc=f'Predicting boxes for {dataset.__class__.__name__}', disable=disable_tqdm)):

            batch_offset = batch_idx * batch_size
            imgs = imgs.to(self.device, non_blocking=True)
            dict_data = {'x': imgs}
            h, w = imgs.shape[-2:]
            dict_outputs: dict = self._forward(dict_data, device=self.device)

            batch_pred_masks: torch.Tensor = dict_outputs["mask_pred"]  # [0, 1]
            batch_objectness: torch.Tensor = dict_outputs.get("objectness", None)  # [0, 1]

            if len(batch_pred_masks.shape) == 5:
                # b x n_layers x n_queries x h x w -> b x n_queries x h x w
                batch_pred_masks = batch_pred_masks[:, -1, ...]  # extract the output from the last decoder layer

                if batch_objectness is not None:
                    # b x n_layers x n_queries x 1 -> b x n_queries x 1
                    batch_objectness = batch_objectness[:, -1, ...]

            # resize prediction to original resolution
            # note: upsampling by 4 and cutting the padded region allows for a better result
            batch_pred_masks = F.interpolate(
                batch_pred_masks, scale_factor=4, mode="bilinear", align_corners=False
            )[..., :h, :w]

            # iterate over batch dimension
            for batch_index, pred_masks in enumerate(batch_pred_masks):

                # n_queries x 1 -> n_queries
                objectness: torch.Tensor = batch_objectness[batch_index].squeeze(dim=-1)
                ranks = torch.argsort(objectness, descending=True)  # n_queries
                pred_mask = pred_masks[ranks[0]].cpu()
                bin_pred_mask = pred_mask > 0.5
                obj_segments[num_batches_in_file*batch_size + batch_index] = bin_pred_mask

                if self.debug:
                    self.visualise(batch_offset+batch_index, imgs[batch_index], dataset.classes[target['classes'][batch_index]][0], bin_pred_mask, plot_dir)

            num_batches_in_file += 1
            if use_multiple_files and num_batches_in_file == max_num_batches_per_file:
                cur_segmentations_file_path = os.path.join(seg_dir, segmentations_file_prefix + f'_{batch_offset-(num_batches_in_file-1)*batch_size}_{batch_offset+imgs.shape[0]-1}'+segmentations_file_postfix)
                with open(cur_segmentations_file_path, 'wb') as f:
                    torch.save(obj_segments[:(num_batches_in_file-1)*batch_size+imgs.shape[0]].clone(), f)
                num_batches_in_file = 0
                obj_segments = torch.zeros((max_num_batches_per_file * batch_size, img_size, img_size), dtype=torch.bool)
                LOGGER.debug(f'Saved file number {(batch_idx+1) // max_num_batches_per_file} of {total_num_files} ({batch_idx} of {len(img_data_loader)} batches)')

            if self.debug:
                break

        if use_multiple_files:
            if num_batches_in_file > 0:
                cur_segmentations_file_path = os.path.join(seg_dir,
                                                           segmentations_file_prefix + f'_{batch_offset - (num_batches_in_file - 1) * batch_size}_{batch_offset + imgs.shape[0]-1}' + segmentations_file_postfix)
                with open(cur_segmentations_file_path, 'wb') as f:
                    torch.save(obj_segments[:(num_batches_in_file-1)*batch_size+imgs.shape[0]].clone(), f)
        else:
            with open(segmentations_file_path, 'wb') as f:
                torch.save(obj_segments, f)
        LOGGER.debug(f'Saved final file')

        return obj_segments


if __name__ == '__main__':
    from argparse import ArgumentParser, Namespace
    import yaml

    base_dir = pathlib.Path(__file__).parent
    def_config_path = os.path.join(base_dir, 'src', 'third_party', 'selfmask', 'configs', 'duts-dino-k234-nq20-224-swav-mocov2-dino-p16-sr10100.yaml')
    def_weights_path = os.path.join(base_dir, 'data', 'selfmask_nq20.pt')

    parser = ArgumentParser("SelfMask evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default=def_config_path
    )

    parser.add_argument(
        "--p_state_dict",
        type=str,
        default=def_weights_path,
    )

    parser.add_argument('--data_dir', default='', type=str, help='Path to the root data directory.')
    parser.add_argument('--processed_dir', default='', type=str, help='Path to the directory for processed data.')
    parser.add_argument('--img_net_path', default='', type=str, help='Path to the ImageNet directory.')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--disable_tqdm', action='store_true', help='Disable TQDM')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers')
    parser.add_argument('--max_num_batches_per_file', default=128, type=int, help='Maximum number of batches per file')
    parser.add_argument('--img_size', default=224, type=int, help='Size of the images')

    # independent variables
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dir_root', type=str, default="..")
    parser.add_argument('--suffix', type=str, default='')
    args: Namespace = parser.parse_args()
    base_args = yaml.safe_load(open(f"{args.config}", 'r'))
    base_args.pop('dataset_name')
    base_args.pop('batch_size')
    base_args.pop('num_workers')
    base_args.pop('use_gpu')
    args: dict = vars(args)
    args.update(base_args)
    args: Namespace = Namespace(**args)
    args.experim_name = 'test'
    dir_ckpt = f"{os.path.dirname(args.p_state_dict)}"

    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    LOGGER.debug(f'Using device: {device}')

    # plot_dir
    results_dir = os.path.join(base_dir, 'results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    # plot_dir
    plot_dir = os.path.join(results_dir, 'plots')
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    # data_dir
    if args.data_dir == '':
        base_data_dir = os.path.join(base_dir, 'data')
    else:
        base_data_dir = args.data_dir

    # set seed
    set_seeds(args.seed)

    state_dict = torch.load(args.p_state_dict, map_location=device)
    model = get_model(arch="maskformer", configs=args).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    LOGGER.info(f"Pre-trained weights are loaded from {args.p_state_dict}.")

    stds = [0.229, 0.224, 0.225]
    means = [0.485, 0.456, 0.406]
    trans, resize_img_size = get_resize_and_cropping_transforms(args.img_size, cropping=False)
    trans += [transforms.Normalize(means, stds)]
    transform = transforms.Compose(trans)
    def denormalise(image_tensor):
        image_tensor *= torch.tensor(stds, device=device).view(3, 1, 1)
        image_tensor += torch.tensor(means, device=device).view(3, 1, 1)
        return image_tensor

    seg_annotator = Annotator(
        network=model,
        device=device,
        denormalise=denormalise
    )

    dataset_name = 'imagenet'
    if dataset_name.lower() == 'imagenet':
        if args.processed_dir == '':
            processed_meta_dir = os.path.join(base_data_dir, 'ImageNet')
        else:
            processed_meta_dir = os.path.join(args.processed_dir, 'ImageNet')
        dataset = CustomImageNet(root=args.img_net_path, processed_meta_dir=processed_meta_dir,
                                 split='train', transform=transform)
        seg_dir = os.path.join(dataset.processed_meta_dir, 'segmentations', 'selfmask')
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)

    else:
        raise ValueError(f'dataset_name {dataset_name} is not supported.')
    dataset_plot_dir = os.path.join(plot_dir, dataset.__class__.__name__)
    if not os.path.exists(dataset_plot_dir):
        os.makedirs(dataset_plot_dir)

    seg_annotator(dataset, seg_dir, dataset_plot_dir, img_size=args.img_size, scale_factor=args.scale_factor, batch_size=args.batch_size,
                  num_workers=args.num_workers, max_num_batches_per_file=args.max_num_batches_per_file, disable_tqdm=args.disable_tqdm)


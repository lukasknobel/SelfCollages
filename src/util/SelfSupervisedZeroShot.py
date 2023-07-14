import logging
import os
from pathlib import Path

import matplotlib.patheffects as PathEffects
import numpy as np
import torch
from IPython.core.display_functions import display
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import label

from . import transforms
from .box_util import get_box_from_binary_mask
from .misc import get_filtered_model_dirs
from .model_loader import load_model
from ..models.backbones import Backbone

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
LOGGER = logging.getLogger()


@torch.no_grad()
def test_self_supervised_zero_shot(model_id, models_dir, device, weights_dir, imgs_path, model_dir=None, img_size=384,
                                   num_iterations=3, show_all_exemplars=True, max_clusters=4,
                                   ign_stop_conditions=False, scale_exemplar_factor=0.7, plot_count=False, notebook=False):

    show_all_exemplars = show_all_exemplars and not notebook

    if model_dir is None or model_dir == '':
        model_dirs, sub_dirs = get_filtered_model_dirs(models_dir, [str(model_id)], return_sub_dirs=True,
                                                       include_baselines=False, check_beginning=True)
        if len(model_dirs) == 0:
            raise FileNotFoundError(f'Error: No model found with id {model_id}')
        if len(model_dirs) > 1:
            LOGGER.warning(f'Warning: Multiple models found with id {model_id} ({model_dirs}). Using the first one.')
        model_dir = model_dirs[0]
        LOGGER.info(f'Overwriting model_dir with {model_dir}')

    # load model
    model, model_args = load_model(model_dir, device, weights_dir)
    model.eval()
    model = model.to(device)

    result_file_prefix = 'pred_'

    model_backbone = Backbone(model.backbone_type, model_args.weights, weights_dir, return_feature_vector=False,
                                 backbone_freezing=model_args.backbone_freezing, num_frozen_blocks=model_args.num_frozen_blocks,
                                 return_cls_self_attention=True, return_vector_and_map=False)
    model_backbone.load_state_dict(model.backbone.state_dict())
    model_backbone = model_backbone.to(device)

    normalise_transform = model.backbone.normalise_transform
    transform = [transforms.Resize((img_size, img_size))]
    if normalise_transform is not None:
        transform.append(normalise_transform)
    transform = transforms.Compose(transform)

    # load example images
    img_paths = [Path(os.path.join(imgs_path, el)) for el in os.listdir(imgs_path)]

    for img_path in img_paths:
        if Path(img_path).name.startswith(result_file_prefix) or not os.path.isfile(img_path):
            continue
        try:
            image = Image.open(img_path)
        except Exception as e:
            LOGGER.warning(f'Error: Could not open image {img_path}')
            continue
        image.load()

        img_tensor = transforms.ToTensor()(image)
        if img_tensor.shape[0] == 4:
            img_tensor = img_tensor[:3]

        model_input = transform(img_tensor).unsqueeze(0).to(device)
        h, w = model_input.shape[-2:]
        w_patches = w // model_backbone.patch_size
        h_patches = h // model_backbone.patch_size

        # encode query image
        encoded_img, self_attention = model_backbone(model_input)
        encoded_img = encoded_img.flatten(2).transpose(1, 2)

        # extract attention map
        orig_attention_map = self_attention.mean(1).view(h_patches, w_patches)

        attention_map = orig_attention_map.clone()
        first_attention_max = attention_map.max()

        densities = []
        norm_counts = []
        exemplars = []

        for _ in range(max_clusters):
            # select exemplar
            if not ign_stop_conditions and attention_map.max()/first_attention_max < 0.2:
                LOGGER.debug('break because remaining attention is too small')
                break

            from torchvision.transforms.functional import gaussian_blur
            attention_map = gaussian_blur(attention_map.unsqueeze(0), 3, 1.5).squeeze(0)

            highest_attention_idx = torch.argmax(attention_map)
            idx_y = highest_attention_idx // h_patches
            idx_x = highest_attention_idx % h_patches

            close_to_max_mask = attention_map > attention_map.max() * 0.5
            connected_components = label(close_to_max_mask.cpu(), structure=np.ones((3, 3)))[0]
            exemplar_region = connected_components == connected_components[idx_y, idx_x]
            exemplar_coords = get_box_from_binary_mask(torch.tensor(exemplar_region))
            exemplar_x_start, exemplar_y_start, exemplar_x_end, exemplar_y_end = exemplar_coords*model_backbone.patch_size

            for i in range(num_iterations):
                exemplar = model_input[:, :, exemplar_y_start:exemplar_y_end, exemplar_x_start:exemplar_x_end]
                exemplar = transforms.Resize((model.reference_img_size, model.reference_img_size))(exemplar).unsqueeze(1)

                # augment exemplars to construct multiple shots
                # if i == num_iterations-1:
                #     flip_h = transforms.RandomHorizontalFlip(p=1)
                #     color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
                #     crop = transforms.Compose([transforms.RandomCrop((int(model.reference_img_size*0.66), int(model.reference_img_size*0.66))), transforms.Resize((model.reference_img_size, model.reference_img_size))])
                #     # exemplar_2 = flip_h(exemplar[0])
                #     exemplar_2 = crop(normalise_transform(color_jitter(normalise_transform.reverse(exemplar[0]))))
                #     exemplar_3 = crop(normalise_transform(color_jitter(normalise_transform.reverse(exemplar[0]))))
                #     exemplar = torch.cat([exemplar, exemplar_2.unsqueeze(0), exemplar_3.unsqueeze(0)], 1)

                # run backbone on exemplar and predict density map
                encoded_exemplar = model.forward_encode_exemplars(encoded_img, None, exemplar, exemplar.shape[1])
                x = model.forward_interaction_module(encoded_img, encoded_exemplar, h, w)
                pred = model.pred_head(x).cpu()

                if len(densities) > 0:
                    pred = torch.clamp(pred - torch.clamp(torch.concat(densities, 0).sum(0), 0), 0)

                pred_cnt = pred.sum() / model_args.density_scaling
                norm_cnt = pred_cnt / (pred[:, exemplar_y_start:exemplar_y_end, exemplar_x_start:exemplar_x_end].sum()/ model_args.density_scaling)

                density_connected_components = label((pred > pred.max() * 0.2)[0], structure=np.ones((3, 3)))[0]

                # get second largest connected component
                component_idx, sizes = torch.tensor(density_connected_components).unique(return_counts=True)
                second_largest_component_idx = component_idx[sizes.argsort()[-2]]
                new_exemplar_mask = density_connected_components == second_largest_component_idx.item()

                exemplar_x_start, exemplar_y_start, exemplar_x_end, exemplar_y_end = get_box_from_binary_mask(
                    torch.tensor(new_exemplar_mask))

                # reduce size
                center_x = (exemplar_x_start + exemplar_x_end) // 2
                center_y = (exemplar_y_start + exemplar_y_end) // 2
                dim_x = exemplar_x_end - exemplar_x_start
                dim_y = exemplar_y_end - exemplar_y_start
                dim_x = int(dim_x * scale_exemplar_factor)
                dim_y = int(dim_y * scale_exemplar_factor)
                exemplar_x_start = center_x - dim_x // 2
                exemplar_x_end = center_x + dim_x // 2
                exemplar_y_start = center_y - dim_y // 2
                exemplar_y_end = center_y + dim_y // 2

                density_map_mask = torch.ones(attention_map.shape, dtype=torch.bool)

                density_map_mask = density_map_mask & transforms.Resize((attention_map.shape[0], attention_map.shape[1]))(pred < 0.5)[0]

                # exclude exemplar region
                density_map_mask[
                exemplar_y_start // model_backbone.patch_size:exemplar_y_end // model_backbone.patch_size,
                exemplar_x_start // model_backbone.patch_size:exemplar_x_end // model_backbone.patch_size] = False

                # exclude max attention region
                mask_offset = 2
                density_map_mask[max(idx_y-mask_offset, 0):min(idx_y+1+mask_offset, h_patches), max(idx_x-mask_offset, 0):min(idx_x+1+mask_offset, w_patches)] = False

                attention_map = attention_map * density_map_mask.to(device)

                if show_all_exemplars:
                    unnorm_exemplars = normalise_transform.reverse(exemplar[0])
                    combined_exemplars = torch.concat([unnorm_exemplar for unnorm_exemplar in unnorm_exemplars], dim=-1).cpu()
                    exemplars.append(combined_exemplars)

            if not show_all_exemplars:
                unnorm_exemplars = normalise_transform.reverse(exemplar[0])
                combined_exemplars = torch.concat([unnorm_exemplar for unnorm_exemplar in unnorm_exemplars], dim=-1).cpu()
                exemplars.append(combined_exemplars)
            densities.append(pred)
            norm_counts.append(round(norm_cnt.item(),2))

        if notebook:
            w, h = image.size
            fontsize = 10
            plt.rcParams.update({'font.size': fontsize})
            num_images = (1 + len(densities))
            plt.figure(figsize=((len(densities) + 1) * 2, 3))
            cell_width = 5
            gs = GridSpec(3, (len(densities) + 1) * cell_width, hspace=0.0)
            axes = []
            for r in range(2):
                row_axes = []
                for c in range(len(densities) + 1):
                    if r == 0:
                        row_axes.append(plt.subplot(gs[r:r + 1, c * cell_width + 1:(c + 1) * cell_width - 1]))
                    else:
                        row_axes.append(plt.subplot(gs[r:, c * cell_width:(c + 1) * cell_width]))
                axes.append(row_axes)

            axes[0][0].axis('off')

            axes[1][0].imshow(image)
            axes[1][0].axis('off')
            density_transform = transforms.Resize((h, w))
            name = f'{result_file_prefix}{img_path.stem}_exemplars'
            LOGGER.info(f'Counts for image {img_path.stem}: {norm_counts}')
            for i, density in enumerate(densities):
                axes[0][1 + i].set_title(f'Type {i + 1}')
                axes[0][1 + i].imshow(exemplars[i].permute(1, 2, 0).clip(0, 1))
                axes[0][1 + i].axis('off')

                pred_count = norm_counts[i]
                density = density_transform(density)[0]
                density /= density.max()
                map_ = torch.stack((torch.zeros_like(density), torch.zeros_like(density), density)).permute(1, 2, 0)
                axes[1][1 + i].imshow((img_tensor.permute(1, 2, 0) * 0.4 + map_ * 1).clip(0, 1))
                str_pred_cnt = str(pred_count)
                txt = axes[1][1 + i].text(w*0.02, h*0.25, str_pred_cnt, c='w', fontsize='xx-large')
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])
                axes[1][1 + i].axis('off')

            plt.tight_layout()
            display(plt.gcf())
            plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})
        else:
            w, h = image.size

            fontsize = int(0.15*h)
            plt.rcParams.update({'font.size': fontsize})
            num_images = (1 + len(densities))
            plt.figure(figsize=(5, 5))
            plt.subplot(2, num_images, 1)
            plt.imshow(image)
            plt.axis('off')

            density_transform = transforms.Resize((h, w))
            name = f'{result_file_prefix}{img_path.stem}_exemplars'

            LOGGER.info(f'Counts for image {img_path.stem}: {norm_counts}')

            for i, density in enumerate(densities):
                plt.figure(figsize=(w / 30, h / 30))
                pred_count = norm_counts[i]

                density = density_transform(density)[0]
                density /= density.max()
                map_ = torch.stack((torch.zeros_like(density), torch.zeros_like(density), density)).permute(1, 2, 0)
                plt.imshow((img_tensor.permute(1, 2, 0) * 0.4 + map_ * 1).clip(0,1))
                if plot_count:
                    str_pred_cnt = str(pred_count)
                    txt = plt.text(10, fontsize+10, str_pred_cnt, c='w', fontsize='xx-large')
                    txt.set_path_effects([PathEffects.withStroke(linewidth=6, foreground='k')])
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(img_path.parent, f'{name}_{i}_density.png'), bbox_inches='tight')

                if show_all_exemplars:
                    for j in range(num_iterations):
                        plt.figure()
                        plt.imshow(exemplars[i*num_iterations+j].permute(1, 2, 0).clip(0,1))
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(os.path.join(img_path.parent, f'{name}_{i}_exemplar_{j}.png'), bbox_inches='tight')
                else:
                    plt.figure()
                    plt.imshow(exemplars[i].permute(1, 2, 0))
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(img_path.parent, f'{name}_{i}_exemplar.png'), bbox_inches='tight')
            plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})

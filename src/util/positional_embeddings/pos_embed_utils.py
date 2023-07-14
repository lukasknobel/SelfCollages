"""
Based on facbooks ViT implementation
https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py
This implementation swaps w and h for the scaling factors.
"""
import math

import torch
from torch import nn


def interpolate_pos_encoding(h, w, num_patches, dim_, pos_embed, patch_size, include_cls_token=False):

    N = pos_embed.shape[1]
    if include_cls_token:
        N = N - 1

    if num_patches == N and w == h:
        return pos_embed
    if include_cls_token:
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
    else:
        patch_pos_embed = pos_embed
    w0 = w // patch_size
    h0 = h // patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim_).permute(0, 3, 1, 2),
        scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
        mode='bicubic',
    )
    assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim_)
    if include_cls_token:
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    else:
        return patch_pos_embed

import logging
import math

import torch.nn.functional as F
from torch import nn

LOGGER = logging.getLogger()


class DensityMapDecoder(nn.Module):

    def __init__(self, input_dim, upsampling_factor, upsampling_dim=256):
        super().__init__()

        self.pred_head = []

        num_upsampling_blocks = math.log2(upsampling_factor)
        self.upsampling_factors = [2] * int(num_upsampling_blocks)

        if num_upsampling_blocks != int(num_upsampling_blocks):
            self.upsampling_factors.append(upsampling_factor / 2 ** int(num_upsampling_blocks))
            LOGGER.warning(f'Upsampling factor {upsampling_factor} is not a power of 2, final upsampling factor will be {self.upsampling_factors[-1]}')

        pred_dims = [input_dim] + [upsampling_dim] * len(self.upsampling_factors)
        for i, (in_c, out_c) in enumerate(zip(pred_dims[:-1], pred_dims[1:])):
            loc_pred_layers = [nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                               nn.GroupNorm(8, out_c),
                               nn.ReLU(inplace=True)]
            if i == len(pred_dims) - 2:
                loc_pred_layers.append(nn.Conv2d(out_c, 1, kernel_size=1, stride=1))
            self.pred_head.append(nn.Sequential(*loc_pred_layers))
        self.pred_head = nn.ModuleList(self.pred_head)

    def forward(self, x):

        # Density map regression
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)

        for layer, upsampling_factor in zip(self.pred_head, self.upsampling_factors):
            x = layer(x)
            upsampling_size = x.shape[-1] * upsampling_factor
            if int(upsampling_size) != upsampling_size:
                raise ValueError(f'Upsampling factor of {upsampling_factor} leads to upsampling size {upsampling_size} which is not an integer.')
            x = F.interpolate(x, size=int(upsampling_size), mode='bilinear', align_corners=False)

        # [N, H, W]
        x = x.squeeze(-3)

        return x

"""
Based on https://github.com/Verg-Avesta/CounTR/blob/5bfacc7f837da10b3a8fc0e677264d62291900a2/models_mae_cross.py
"""
import logging

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.ops import roi_align

from .backbones import Backbone, BackboneFreezing
from .modules import get_head, CrossAttentionBlock
from ..util import Constants, plotting
from ..util.positional_embeddings import interpolate_pos_encoding, get_2d_sincos_pos_embed

LOGGER = logging.getLogger()


class UnCounTRModel(nn.Module):
    def __init__(self, args, weights_dir=None, fim_embed_dim=512, fim_depth=2, decoder_num_heads=16, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        LOGGER.info(f'Ignoring the following arguments: {kwargs}')

        self.backbone_type = args.backbone_type
        self.frozen_encoder = args.backbone_freezing is BackboneFreezing.completely
        self.reference_img_size = args.reference_crop_size
        self.share_encoder = args.share_encoder
        self.fim_embed_dim = fim_embed_dim
        self.use_exemplar_attention = args.use_exemplar_attention
        self.use_exemplar_roi_align = args.use_exemplar_roi_align
        self.use_exemplar_cls_token = args.use_exemplar_cls_token
        self.split_map_and_count = args.split_map_and_count
        self.weigh_by_similarity = args.weigh_by_similarity
        self.use_similarity_projection = args.use_similarity_projection
        self.unified_fim = args.unified_fim
        self.return_backbone_layer_features = args.return_backbone_layer_features

        if not self.share_encoder and self.use_exemplar_cls_token:
            raise ValueError(f'Exemplar cls token is only supported when sharing the encoder.')
        if self.weigh_by_similarity and self.use_similarity_projection:
            raise ValueError(f'Cannot combine multiple similarity weighting methods.')

        # are the encoded exemplar and query image in the same feature space?
        self.shared_feature_space = self.share_encoder or self.use_exemplar_roi_align

        if self.unified_fim and self.use_exemplar_attention:
            raise ValueError(f'Unified fim and exemplar attention are mutually exclusive.')
        if self.unified_fim and not self.shared_feature_space:
            raise ValueError(f'Unified fim is only supported when query image and exemplars share the same feature space.')

        if (self.weigh_by_similarity or self.use_similarity_projection) and not self.shared_feature_space:
            raise ValueError(f'Cannot weigh by similarity if the feature spaces are not shared.')

        self.backbone_att = self.share_encoder and not self.use_exemplar_cls_token

        # encoder
        self.backbone = Backbone(self.backbone_type, args.weights, weights_dir, return_feature_vector=False,
                                 backbone_freezing=args.backbone_freezing, num_frozen_blocks=args.num_frozen_blocks,
                                 return_cls_self_attention=self.backbone_att, return_vector_and_map=self.use_exemplar_cls_token,
                                 return_backbone_layer_features=self.return_backbone_layer_features)
        self.encoder = self.backbone

        if not hasattr(self.encoder, 'patch_size'):
            raise ValueError('Patch size is not defined for the backbone')
        self.patch_size = self.encoder.patch_size

        self.num_patches = (args.img_size // self.patch_size) ** 2

        # exemplar encoder
        if self.share_encoder and self.use_exemplar_roi_align:
            LOGGER.warning('Specified share_encoder and use_exemplar_roi_align. Ignoring use_exemplar_roi_align.')
        if self.share_encoder:
            self.exemplar_encoder = self.encoder
        elif self.use_exemplar_roi_align:
            self.exemplar_encoder = None
        else:
            self.exemplar_encoder = nn.Sequential()
            # [3,64,64]->[64,32,32]->[128,16,16]->[256,8,8]->[decoder_embed_dim,1,1]
            n_channels = [3, 64, 128, 256, self.fim_embed_dim]
            for i, (in_c, out_c) in enumerate(zip(n_channels[:-1], n_channels[1:])):
                self.exemplar_encoder.add_module(f'conv{i}', nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1))
                self.exemplar_encoder.add_module(f'norm{i}', nn.InstanceNorm2d(out_c))
                self.exemplar_encoder.add_module(f'relu{i}', nn.ReLU(inplace=True))

                if i == len(n_channels) - 2:
                    self.exemplar_encoder.add_module(f'pool{i}', nn.AdaptiveAvgPool2d((1, 1)))
                else:
                    self.exemplar_encoder.add_module(f'pool{i}', nn.MaxPool2d(2))
        # zero-shot token
        if self.shared_feature_space:
            self.shot_token = nn.Parameter(torch.zeros(self.backbone.num_features))
        else:
            self.shot_token = nn.Parameter(torch.zeros(self.fim_embed_dim))
        # exemplar padding
        self.exemplar_pad_token = None
        self.max_shots = args.max_shots
        if args.vary_shots_per_sample:
            if self.shared_feature_space:
                self.exemplar_pad_token = nn.Parameter(torch.zeros(self.backbone.num_features))
            else:
                self.exemplar_pad_token = nn.Parameter(torch.zeros(self.fim_embed_dim))

        # feature interaction module
        self.fim_embed = nn.Linear(self.encoder.num_features, self.fim_embed_dim, bias=True)
        self.fim_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.fim_embed_dim), requires_grad=False)
        self.fim_blocks = None
        self.fim_exemplar_blocks = None
        self.exemplar_embed = None
        if self.unified_fim:
            self.fim_blocks = nn.MultiheadAttention(fim_embed_dim, decoder_num_heads, batch_first=True)
            self.exemplar_embed = nn.Parameter(torch.zeros(1, 1, self.fim_embed_dim))
        else:
            self.fim_blocks = nn.ModuleList([
                CrossAttentionBlock(fim_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                    norm_layer=norm_layer)
                for _ in range(fim_depth)])
            self.fim_exemplar_blocks = None
            if self.use_exemplar_attention:
                self.fim_exemplar_blocks = nn.ModuleList([
                    CrossAttentionBlock(fim_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                        norm_layer=norm_layer, use_self_attention=False, gated_cross_attention=True)
                    for _ in range(fim_depth)])

        self.sim_projection = None
        if self.use_similarity_projection:
            self.sim_projection = nn.Linear(self.encoder.num_features, self.fim_embed_dim)

        self.fim_norm = norm_layer(fim_embed_dim)
        # add cls token if not using density map or split map and count
        self.fim_cls_token = None
        if self.split_map_and_count:
            self.fim_cls_token = nn.Parameter(torch.zeros(1, 1, self.fim_embed_dim))

        # prediction head
        self.pred_head = get_head(args, self.fim_embed_dim, upsampling_factor=self.patch_size)

        self.initialize_weights()

        self.backbone_modules = nn.ModuleList([self.backbone])
        self.counting_modules = nn.ModuleList([self.fim_embed, self.fim_blocks, self.fim_norm, self.fim_exemplar_blocks,
                                               self.pred_head, self.sim_projection])
        if not args.share_encoder:
            # if we share the encoder, we already added it using self.backbone
            self.counting_modules.append(self.exemplar_encoder)

        self.backbone_parameters = nn.ParameterList(list(self.backbone_modules.parameters()))
        self.counting_parameters = nn.ParameterList(list(self.counting_modules.parameters()) +
                                                    [self.fim_pos_embed, self.shot_token, self.fim_cls_token,
                                                     self.exemplar_embed, self.exemplar_pad_token])

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
        #                                     cls_token=False)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        fim_pos_embed = get_2d_sincos_pos_embed(self.fim_pos_embed.shape[-1],
                                                int(self.num_patches ** .5), cls_token=False)
        self.fim_pos_embed.data.copy_(torch.from_numpy(fim_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.shot_token, std=.02)
        if self.fim_cls_token is not None:
            torch.nn.init.normal_(self.fim_cls_token, std=.02)
        if self.exemplar_embed is not None:
            torch.nn.init.normal_(self.exemplar_embed, std=.02)
        if self.exemplar_pad_token is not None:
            torch.nn.init.normal_(self.exemplar_pad_token, std=.02)

        # initialize the model, ignoring the backbone
        for name, module in self.named_modules():
            if 'backbone' in name:
                continue
            self._init_weights(module)

    def _init_weights(self, m):
        # initialize nn.Linear and nn.LayerNorm
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encode_exemplars(self, encoded_img, exemplar_boxes, exemplar_crops, shot_num):

        if isinstance(shot_num, int):
            shot_num = torch.full((encoded_img.shape[0],), shot_num, device=encoded_img.device)
        elif self.exemplar_pad_token is None:
            raise ValueError('shot_num is a Tensor, but exemplar_pad_token is None')

        cur_max_shot = shot_num.max().item()
        mask = None
        if self.exemplar_pad_token is not None:
            # pad exemplars to max_shots
            if cur_max_shot > self.max_shots:
                LOGGER.warning(f'Using up to {cur_max_shot} shots, but the model was only trained with max_shots={self.max_shots}')
                mask = torch.arange(cur_max_shot, device=shot_num.device).unsqueeze(0) < shot_num.unsqueeze(1)
            else:
                mask = torch.arange(self.max_shots, device=shot_num.device).unsqueeze(0) < shot_num.unsqueeze(1)

        if cur_max_shot == 0:
            # return zero-shot token if the maximum shot_num is 0
            encoded_exemplars = self.shot_token.repeat(exemplar_crops.shape[0], 1, 1)
        elif self.use_exemplar_roi_align:
            n_patches_w = int(encoded_img.shape[1] ** 0.5)
            reshaped_img = encoded_img.view(encoded_img.shape[0], n_patches_w, n_patches_w, -1).permute(0, 3, 1, 2)
            modified_boxes = torch.concat(
                [torch.arange(exemplar_boxes.shape[0], device=exemplar_boxes.device).repeat_interleave(cur_max_shot).unsqueeze(1),
                 exemplar_boxes[:, :cur_max_shot].reshape(-1, exemplar_boxes.shape[-1])], dim=-1)
            encoded_exemplars = roi_align(reshaped_img, modified_boxes, output_size=(1, 1), spatial_scale=n_patches_w, aligned=True)
            encoded_exemplars = encoded_exemplars.squeeze(-1).squeeze(-1)

            # [N*cur_max_shot, C] -> [N, cur_max_shot, C]
            encoded_exemplars = encoded_exemplars.view(-1, cur_max_shot, encoded_exemplars.shape[-1])
        else:
            # [N,max_exemplars,3,exemplar_H,exemplar_W]->[N,cur_max_shot,3,exemplar_H,exemplar_W]
            exemplar_crops = exemplar_crops[:, :cur_max_shot]
            # [N,cur_max_shot,3,exemplar_H,exemplar_W]->[N*cur_max_shot,3,exemplar_H,exemplar_W]
            exemplar_crops = exemplar_crops.reshape(-1, *exemplar_crops.shape[2:])
            exemplar_crops = self.exemplar_encoder(exemplar_crops)
            if self.backbone_att:
                encoded_exemplars = (exemplar_crops[0].flatten(-2) * exemplar_crops[1].mean(1, keepdims=True)).sum(-1)
            elif self.use_exemplar_cls_token:
                encoded_exemplars = exemplar_crops[0]
            else:
                encoded_exemplars = exemplar_crops.squeeze(-1).squeeze(-1)
            # [N*cur_max_shotC]->[N,cur_max_shot,C]
            encoded_exemplars = encoded_exemplars.view(-1, cur_max_shot, encoded_exemplars.shape[-1])  # [shot_num,N,C]->[N,shot_num,C] # [shot_num,N,C]->[N,shot_num,C]

        if mask is not None:
            # pad masked exemplars with pad token
            pad_token = self.exemplar_pad_token.unsqueeze(0).unsqueeze(0).repeat(encoded_exemplars.shape[0], 1, 1)
            # pad everything to max_shots
            padded_encoded_exemplars = pad_token.repeat(1,mask.shape[1],1)
            padded_encoded_exemplars[:, :encoded_exemplars.shape[1]] = encoded_exemplars
            # replace individual exemplars with pad token
            encoded_exemplars = torch.where(mask.unsqueeze(-1), padded_encoded_exemplars, pad_token)
            # replace exemplars with zero-shot token
            zero_shot = shot_num == 0
            encoded_exemplars[zero_shot, 0] = self.shot_token

        return encoded_exemplars

    def forward_interaction_module(self, x, encoded_exemplars, h, w):

        if self.weigh_by_similarity:
            min_dist_query_patch_any_exemplar = torch.cdist(x, encoded_exemplars).min(-1)[0]
            glob_min_dist = min_dist_query_patch_any_exemplar.min(1)[0].unsqueeze(-1).repeat(1, x.shape[1])
            glob_max_dist = min_dist_query_patch_any_exemplar.max(1)[0].unsqueeze(-1).repeat(1, x.shape[1])
            norm_dist = (min_dist_query_patch_any_exemplar - glob_min_dist) / (glob_max_dist - glob_min_dist)
            norm_similarity = 1 - norm_dist
        elif self.use_similarity_projection:
            elementwise_difference = (x.unsqueeze(-1) - encoded_exemplars.permute(0, 2, 1).unsqueeze(1).repeat(1, x.shape[1], 1, 1)).abs()
            elementwise_min_difference = elementwise_difference.min(-1)[0]
            sample_min_diff = elementwise_min_difference.min(-1)[0].min(-1)[0]
            sample_max_diff = elementwise_min_difference.max(-1)[0].max(-1)[0]
            sample_min_diff = sample_min_diff.unsqueeze(-1).unsqueeze(-1).repeat(1, *x.shape[1:])
            sample_max_diff = sample_max_diff.unsqueeze(-1).unsqueeze(-1).repeat(1, *x.shape[1:])
            norm_elementwise_dist = (elementwise_min_difference - sample_min_diff) / (sample_max_diff - sample_min_diff)
            norm_elementwise_similarity = 1 - norm_elementwise_dist
            projected_similarity = self.sim_projection(norm_elementwise_similarity)
        # embed tokens
        x = self.fim_embed(x)
        if self.shared_feature_space:
            encoded_exemplars = self.fim_embed(encoded_exemplars)

        # apply similarity weighting
        if self.weigh_by_similarity:
            x = x * norm_similarity.unsqueeze(-1)
        elif self.use_similarity_projection:
            x = x + projected_similarity

        _, num_patches, dim_ = x.shape

        # add pos embed
        x = x + interpolate_pos_encoding(h, w, num_patches, dim_, self.fim_pos_embed, self.patch_size)

        # add cls token
        if self.split_map_and_count:
            x = torch.cat((self.fim_cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # apply transformer blocks
        if self.unified_fim:
            num_query_patches = x.shape[1]
            encoded_exemplars = encoded_exemplars + self.exemplar_embed
            x = torch.cat((x, encoded_exemplars), dim=1)
            x, _ = self.fim_blocks(x, x, x, need_weights=False)
            x = x[:, :num_query_patches]
        elif self.use_exemplar_attention:
            for exemplar_blk, blk in zip(self.fim_exemplar_blocks, self.fim_blocks):
                encoded_exemplars = exemplar_blk(encoded_exemplars, x)
                x = blk(x, encoded_exemplars)
        else:
            for blk in self.fim_blocks:
                x = blk(x, encoded_exemplars)
        x = self.fim_norm(x)

        return x

    def forward(self, imgs, exemplar_boxes, exemplar_crops, shot_num):
        if self.frozen_encoder:
            with torch.no_grad():
                latent = self.encoder(imgs)
        else:
            latent = self.encoder(imgs)
        # [N, num_patches, self.encoder.num_features]
        if self.backbone_att:
            # remove attention
            latent, _ = latent
        elif self.use_exemplar_cls_token:
            # remove cls token
            _, latent = latent
        latent = latent.flatten(2).transpose(1, 2)

        # exemplar encoder
        encoded_exemplars = self.forward_encode_exemplars(latent, exemplar_boxes, exemplar_crops, shot_num)

        _, _, h, w = imgs.shape

        # feature interaction
        x = self.forward_interaction_module(latent, encoded_exemplars, h, w)

        # decoder
        pred = self.pred_head(x)

        return pred

    def __repr__(self):
        return f'{self.__class__.__name__}'

    @torch.no_grad()
    def visualise_prediction(self, imgs, target_dicts, shot_num, device, denormalise=None, density_scaling=1.0,
                             return_fig=False, **kwargs):

        preds = self(imgs.to(device), target_dicts[Constants.TARGET_DICT_BOXES].to(device), target_dicts[Constants.TARGET_DICT_REF_IMGS].to(device), shot_num)
        preds = preds.cpu()
        if shot_num > 0:
            gt_cnts = list(target_dicts[Constants.TARGET_DICT_GLOBAL_COUNT].cpu().numpy())
        else:
            gt_cnts = list(target_dicts[Constants.TARGET_DICT_TOTAL_NUM_OBJECTS].cpu().numpy())

        pred_cnts = list((preds.sum(dim=(-2, -1)) / density_scaling).numpy())

        if denormalise is None:
            plot_imgs = imgs
        else:
            plot_imgs = [denormalise(img) for img in imgs]

        num_samples = len(imgs)
        num_cols = 3
        img_size = 4

        fig = plt.figure(figsize=(num_cols * img_size, (img_size + 0.2)*num_samples))
        for i in range(num_samples):
            ax = plt.subplot(num_samples, num_cols, 1+i*num_cols)
            plt.imshow(plot_imgs[i].permute(1, 2, 0))
            if shot_num > 0:
                boxes = target_dicts[Constants.TARGET_DICT_BOXES][i][:shot_num].clone()
                boxes[:, [0, 2]] *= imgs[i].shape[-1]
                boxes[:, [1, 3]] *= imgs[i].shape[-2]
                plotting.plot_boxes(boxes, fig, ax=ax)
            plt.title(f'GT count: {gt_cnts[i]}')
            plt.axis('off')
            plt.subplot(num_samples, num_cols, 2+i*num_cols)
            plt.imshow(plot_imgs[i].permute(1, 2, 0))
            if shot_num > 0:
                if Constants.TARGET_DICT_DENSITY_MAPS in target_dicts:
                    gt_density = target_dicts[Constants.TARGET_DICT_DENSITY_MAPS][i].unsqueeze(0)
                    plt.imshow(gt_density.permute(1, 2, 0), alpha=0.8, cmap='jet')
                    plt.title(f'GT density-based count: {gt_density.sum().item() / density_scaling:.2f}')
            else:
                if Constants.TARGET_DICT_ALL_DENSITY_MAPS in target_dicts:
                    gt_density = target_dicts[Constants.TARGET_DICT_ALL_DENSITY_MAPS][i].unsqueeze(0)
                    plt.imshow(gt_density.permute(1, 2, 0), alpha=0.8, cmap='jet')
                    plt.title(f'GT density-based count: {gt_density.sum().item() / density_scaling:.2f}')
            plt.axis('off')
            plt.subplot(num_samples, num_cols, 3+i*num_cols)
            plt.imshow(plot_imgs[i].permute(1, 2, 0))
            plt.imshow(preds[i].unsqueeze(0).permute(1, 2, 0), alpha=0.8, cmap='jet')
            plt.title(f'Pred count: {pred_cnts[i]:.2f}')
            plt.axis('off')
            plt.tight_layout()
        if return_fig:
            return fig
        else:
            plt.show()

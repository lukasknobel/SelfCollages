import enum
import logging
import os

import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

from .ModelEnums import PretrainedWeights, BackboneTypes
from ..util import transforms
from ..util.positional_embeddings import interpolate_pos_encoding

LOGGER = logging.getLogger()


class BackboneFreezing(enum.Enum):
    completely = 0
    nothing = 1
    block = 2


class Backbone(nn.Module):

    def __init__(self, backbone_type: BackboneTypes, weights, weights_dir, return_feature_vector,
                 return_cls_self_attention, backbone_freezing, return_vector_and_map=False, num_frozen_blocks=10,
                 return_backbone_layer_features=-1):
        super().__init__()
        self.model = None
        self.num_features = None
        self.num_channels = None
        self.normalise_transform = None
        self.is_vit = False
        self.patch_size = None
        self.num_heads=None
        use_backbone_layer_features = return_backbone_layer_features >= 0

        if use_backbone_layer_features and weights is not PretrainedWeights.DINOv2:
            raise NotImplementedError('Backbone layer features are only supported for DINOv2 weights')
        # ViT backbone
        try:
            self.patch_size = int(backbone_type.name.split('_')[-1])
        except ValueError:
            raise ValueError(f'Could not determine patch size for ViT backbone type {backbone_type.name}')
        LOGGER.debug(f'Using ViT backbone with patch size {self.patch_size}')
        self.is_vit = True
        if weights is PretrainedWeights.DINO:
            self.normalise_transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            if backbone_type is BackboneTypes.ViT_B_8:
                self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8', verbose=False)
            elif backbone_type is BackboneTypes.ViT_B_16:
                self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', verbose=False)
            elif backbone_type is BackboneTypes.ViT_S_16:
                self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', verbose=False)

            if self.model is not None:
                self.num_features = self.model.num_features
                self.num_heads = self.model.blocks[-1].attn.num_heads

                patch_embed_patch_size = self.model.patch_embed.patch_size
                if isinstance(patch_embed_patch_size, tuple):
                    patch_embed_patch_size = patch_embed_patch_size[0]
                if self.patch_size != patch_embed_patch_size:
                    raise ValueError(f'Backbone patch size ({self.patch_size}) != DINO patch size ({self.model.patch_embed.patch_size})')

                # modify model forward
                mod_forward = None
                if return_vector_and_map:
                    def forward(x):
                        self_ = self.model
                        B, nc, h, w = x.shape
                        num_patches_w = w // self.patch_size
                        num_patches_h = h // self.patch_size

                        x = self_.prepare_tokens(x)
                        for blk in self_.blocks:
                            x = blk(x)
                        x = self_.norm(x)

                        # ignore classifier "token"
                        x_map = x[:, 1:]

                        # reshape feature map: (B, num_features, num_patches, num_patches)
                        return x[:, 0], x_map.permute(0, 2, 1).view(x_map.shape[0], x_map.shape[-1], num_patches_h, num_patches_w)

                    mod_forward = forward
                elif not return_feature_vector:

                    if return_cls_self_attention:
                        def forward(x):
                            self_ = self.model
                            B, nc, h, w = x.shape
                            num_patches_w = w // self.patch_size
                            num_patches_h = h // self.patch_size

                            x = self_.prepare_tokens(x)
                            for i, blk in enumerate(self_.blocks):
                                if i < len(self_.blocks) - 1:
                                    x = blk(x)
                                else:
                                    # return attention of the last block
                                    x, att = blk(x), blk(x, return_attention=True)
                                    att = att[:, :, 0, 1:]
                            x = self_.norm(x)

                            # ignore classifier "token"
                            x = x[:, 1:]

                            # reshape feature map: (B, num_features, num_patches, num_patches)
                            return x.permute(0, 2, 1).view(x.shape[0], x.shape[-1], num_patches_h, num_patches_w), att

                    else:
                        def forward(x):
                            self_ = self.model
                            B, nc, h, w = x.shape
                            num_patches_w = w // self.patch_size
                            num_patches_h = h // self.patch_size

                            x = self_.prepare_tokens(x)
                            for blk in self_.blocks:
                                x = blk(x)
                            x = self_.norm(x)

                            # ignore classifier "token"
                            x = x[:, 1:]

                            # reshape feature map: (B, num_features, num_patches, num_patches)
                            return x.permute(0, 2, 1).view(x.shape[0], x.shape[-1], num_patches_h, num_patches_w)
                    mod_forward = forward
                if mod_forward is not None:
                    self.model.forward = mod_forward
                    self.num_channels = self.model.embed_dim
        elif weights is PretrainedWeights.DINOv2:
            self.normalise_transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

            if backbone_type is BackboneTypes.ViT_S_14:
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
            elif backbone_type is BackboneTypes.ViT_B_14:
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
            elif backbone_type is BackboneTypes.ViT_L_14:
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', verbose=False)
            elif backbone_type is BackboneTypes.ViT_G_14:
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', verbose=False)

            if self.model is not None:
                if use_backbone_layer_features:
                    self.num_features = 2 * self.model.num_features
                else:
                    self.num_features = self.model.num_features
                self.num_heads = self.model.blocks[-1].attn.num_heads

                patch_embed_patch_size = self.model.patch_embed.patch_size
                if isinstance(patch_embed_patch_size, tuple):
                    patch_embed_patch_size = patch_embed_patch_size[0]
                if self.patch_size != patch_embed_patch_size:
                    raise ValueError(
                        f'Backbone patch size ({self.patch_size}) != DINO patch size ({self.model.patch_embed.patch_size})')

                if return_cls_self_attention:
                    last_block = self.model.blocks[-1]
                    def mod_attn_block_forward(x):
                        self_ = last_block.attn
                        B, N, C = x.shape
                        qkv = self_.qkv(x).reshape(B, N, 3, self_.num_heads, C // self_.num_heads).permute(2, 0, 3, 1, 4)

                        q, k, v = qkv[0] * self_.scale, qkv[1], qkv[2]
                        attn = q @ k.transpose(-2, -1)

                        attn = attn.softmax(dim=-1)
                        attn = self_.attn_drop(attn)

                        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                        x = self_.proj(x)
                        x = self_.proj_drop(x)
                        return x, attn

                    last_block.attn.forward = mod_attn_block_forward
                    def mod_block_forward(x):
                        self_ = last_block
                        if not isinstance(x, torch.Tensor):
                            raise NotImplementedError('return_cls_self_attention is only implemented for torch.Tensor inputs')

                        def attn_residual_func(x):
                            x, att = self_.attn(self_.norm1(x))
                            return self_.ls1(x), att

                        def ffn_residual_func(x):
                            return self_.ls2(self_.mlp(self_.norm2(x)))

                        if self_.training and self_.sample_drop_ratio > 0.0:
                           raise NotImplementedError('return_cls_self_attention is not implemented for training with sample_drop_ratio > 0.0')
                        else:
                            mod_x, att = attn_residual_func(x)
                            x = x + mod_x
                            x = x + ffn_residual_func(x)
                        return x, att

                    last_block.forward = mod_block_forward

                def mod_forward_features(x, masks=None):
                    self_ = self.model
                    if isinstance(x, list):
                        return self_.forward_features_list(x, masks)

                    x = self_.prepare_tokens_with_masks(x, masks)

                    inter_features = None
                    att = None

                    for i, blk in enumerate(self_.blocks):
                        if return_cls_self_attention and i == len(self_.blocks) - 1:
                            # return attention of the last block
                            x, att = blk(x)
                            att = att[:, :, 0, 1:]
                        else:
                            x = blk(x)
                        if use_backbone_layer_features and i == return_backbone_layer_features:
                            inter_features = x

                    x_norm = self_.norm(x)
                    return {
                        "x_norm_clstoken": x_norm[:, 0],
                        "x_norm_patchtokens": x_norm[:, 1:],
                        "x_prenorm": x,
                        "masks": masks,
                        "att": att,
                        "inter_clstoken": inter_features[:, 0] if inter_features is not None else None,
                        "inter_features": inter_features[:, 1:] if inter_features is not None else None
                    }

                self.model.forward_features = mod_forward_features

                # modify model forward
                def mod_forward(x, *args, is_training=False, **kwargs):
                    self_ = self.model
                    B, nc, h, w = x.shape
                    num_patches_w = w // self.patch_size
                    num_patches_h = h // self.patch_size

                    ret = self_.forward_features(x, *args, **kwargs)

                    feat_vector = ret["x_norm_clstoken"]
                    B, num_patches, num_features = ret["x_norm_patchtokens"].shape
                    # reshape feature map: (B, num_features, num_patches, num_patches)
                    f_map = ret["x_norm_patchtokens"].permute(0, 2, 1).view(B, num_features, num_patches_h,
                                                                            num_patches_w)
                    if use_backbone_layer_features:
                        feat_vector = torch.cat([feat_vector, ret["inter_clstoken"]], dim=1)

                        f_map_inter = ret["inter_features"].permute(0, 2, 1).view(B, num_features, num_patches_h,
                                                                                  num_patches_w)
                        f_map = torch.cat([f_map, f_map_inter], dim=1)

                    if return_feature_vector:
                        return feat_vector
                    else:
                        if return_vector_and_map:
                            return feat_vector, f_map
                        elif return_cls_self_attention:
                            return f_map, ret["att"]
                        else:
                            # only feature map
                            return f_map

                self.model.forward = mod_forward
                self.num_channels = self.model.embed_dim

        elif weights is PretrainedWeights.LEOPART:
            if weights_dir is None:
                raise ValueError(f'weights_dir must be specified for {weights.name} pretrained weights')
            self.normalise_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch8_224, checkpoint_seq

            weights_file_name = None
            if backbone_type is BackboneTypes.ViT_S_16:
                weights_file_name = 'leopart_vits16.ckpt'
                self.model = vit_small_patch16_224()
            elif backbone_type is BackboneTypes.ViT_B_8:
                weights_file_name = 'leopart_vitb8.ckpt'
                self.model = vit_base_patch8_224()

            if self.model is not None:
                path_to_checkpoint = os.path.join(weights_dir, weights_file_name)
                state_dict = torch.load(path_to_checkpoint)
                self.model.load_state_dict({".".join(k.split(".")[1:]): v for k, v in state_dict.items()}, strict=False)

                self.num_features = self.model.num_features
                self.num_heads = self.model.blocks[-1].attn.num_heads
                self.num_channels = self.model.embed_dim

                # modify patch handling to support different resolutions
                # modify patch_embed to remove resolution checks
                model_patch_embed = self.model.patch_embed
                def patch_embed_forward(x):
                    self_ = model_patch_embed
                    x = self_.proj(x)
                    if self_.flatten:
                        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
                    x = self_.norm(x)
                    return x
                model_patch_embed.forward = patch_embed_forward
                # modify position embedding to interpolate for non-native resolutions
                def _pos_embed(x, h, w):
                    self_ = self.model

                    num_patches = x.shape[1]
                    if self_.no_embed_class:
                        # deit-3, updated JAX (big vision)
                        # position embedding does not overlap with class token, add then concat
                        inter_pos_embed = interpolate_pos_encoding(h, w, num_patches, self_.pos_embed.shape[-1],
                                                                   self_.pos_embed,
                                                                   self.patch_size)
                        x = x + inter_pos_embed
                        if self_.cls_token is not None:
                            x = torch.cat((self_.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                    else:
                        # original timm, JAX, and deit vit impl
                        # pos_embed has entry for class token, concat then add
                        if self_.cls_token is not None:
                            x = torch.cat((self_.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                        inter_pos_embed = interpolate_pos_encoding(h, w, num_patches, self_.pos_embed.shape[-1],
                                                                   self_.pos_embed,
                                                                   self.patch_size, include_cls_token=True)
                        x = x + inter_pos_embed
                    return self_.pos_drop(x)
                self.model._pos_embed = _pos_embed

                # modify model forward
                mod_forward = None
                if return_vector_and_map:
                    def forward(x):
                        self_ = self.model
                        B, nc, h, w = x.shape
                        num_patches_w = w // self.patch_size
                        num_patches_h = h // self.patch_size

                        x = self_.patch_embed(x)
                        x = self_._pos_embed(x, h, w)
                        x = self_.norm_pre(x)
                        if self_.grad_checkpointing and not torch.jit.is_scripting():
                            x = checkpoint_seq(self_.blocks, x)
                        else:
                            x = self_.blocks(x)
                        x = self_.norm(x)

                        # ignore classifier "token"
                        x_map = x[:, 1:]

                        # reshape feature map: (B, num_features, num_patches, num_patches)
                        return x[:, 0], x_map.permute(0, 2, 1).view(x_map.shape[0], x_map.shape[-1], num_patches_h,
                                                                    num_patches_w)

                    mod_forward = forward
                elif not return_feature_vector:

                    if return_cls_self_attention:
                        # monkey patch forward of final attention block to return CLS self-attention
                        final_att_block_attn = self.model.blocks[-1].attn
                        def att_forward(x):
                            self_ = final_att_block_attn
                            B, N, C = x.shape
                            qkv = self_.qkv(x).reshape(B, N, 3, self_.num_heads, C // self_.num_heads).permute(2, 0, 3,
                                                                                                            1, 4)
                            q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

                            attn = (q @ k.transpose(-2, -1)) * self_.scale
                            attn = attn.softmax(dim=-1)
                            attn = self_.attn_drop(attn)

                            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                            x = self_.proj(x)
                            x = self_.proj_drop(x)
                            return x, attn
                        final_att_block_attn.forward = att_forward
                        # monkey patch Block module
                        final_att_block = self.model.blocks[-1]
                        def block_forward(x):
                            self_ = final_att_block
                            mod_x, att = self_.attn(self_.norm1(x))
                            x = x + self_.drop_path1(self_.ls1(mod_x))
                            x = x + self_.drop_path2(self_.ls2(self_.mlp(self_.norm2(x))))
                            return x, att
                        final_att_block.forward = block_forward

                        def forward(x):
                            self_ = self.model
                            B, nc, h, w = x.shape
                            num_patches_w = w // self.patch_size
                            num_patches_h = h // self.patch_size

                            x = self_.patch_embed(x)
                            x = self_._pos_embed(x, h, w)
                            x = self_.norm_pre(x)
                            if self_.grad_checkpointing and not torch.jit.is_scripting():
                                x = checkpoint_seq(self_.blocks, x)
                            else:
                                x, att = self_.blocks(x)
                                att = att[:, :, 0, 1:]
                            x = self_.norm(x)

                            # ignore classifier "token"
                            x = x[:, 1:]

                            # reshape feature map: (B, num_features, num_patches, num_patches)
                            return x.permute(0, 2, 1).view(x.shape[0], x.shape[-1], num_patches_h,
                                                           num_patches_w), att

                    else:
                        def forward(x):
                            self_ = self.model

                            B, nc, h, w = x.shape
                            num_patches_w = w // self.patch_size
                            num_patches_h = h // self.patch_size

                            x = self_.patch_embed(x)
                            x = self_._pos_embed(x, h, w)
                            x = self_.norm_pre(x)
                            if self_.grad_checkpointing and not torch.jit.is_scripting():
                                x = checkpoint_seq(self_.blocks, x)
                            else:
                                x = self_.blocks(x)
                            x = self_.norm(x)

                            # ignore classifier "token"
                            x = x[:, 1:]

                            # reshape feature map: (B, num_features, num_patches, num_patches)
                            return x.permute(0, 2, 1).view(x.shape[0], x.shape[-1], num_patches_h, num_patches_w)
                    mod_forward = forward
                else:
                    # return feature vector
                    def forward(x):
                        self_ = self.model

                        B, nc, h, w = x.shape

                        x = self_.patch_embed(x)
                        x = self_._pos_embed(x, h, w)
                        x = self_.norm_pre(x)
                        if self_.grad_checkpointing and not torch.jit.is_scripting():
                            x = checkpoint_seq(self_.blocks, x)
                        else:
                            x = self_.blocks(x)
                        x = self_.norm(x)

                        # return classifier "token"
                        x = x[:, 0]

                        return x
                    mod_forward = forward

                if mod_forward is not None:
                    self.model.forward = mod_forward

        elif weights is PretrainedWeights.IMAGENET or weights is PretrainedWeights.RANDOM:
            if return_cls_self_attention:
                raise NotImplementedError(f'return_cls_self_attention not implemented for {weights.name} weights')
            if backbone_type is BackboneTypes.ViT_B_16:
                if weights is PretrainedWeights.IMAGENET:
                    self.normalise_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
                elif weights is PretrainedWeights.RANDOM:
                    self.model = vit_b_16()
                if self.model is not None:
                    self.num_features = self.model.heads.head.in_features
                    self.model.heads = nn.Identity()
                    self.num_heads = self.model.encoder.layers[-1].num_heads

                    # modify model forward
                    mod_forward = None
                    if return_vector_and_map:
                        def forward(x):
                            self_ = self.model
                            B, nc, h, w = x.shape
                            num_patches_w = w // self_.patch_size
                            num_patches_h = h // self_.patch_size

                            # Reshape and permute the input tensor
                            x = self_._process_input(x)
                            n = x.shape[0]

                            # Expand the class token to the full batch
                            batch_class_token = self_.class_token.expand(n, -1, -1)
                            x = torch.cat([batch_class_token, x], dim=1)

                            x = self_.encoder(x)

                            # ignore classifier "token"
                            x_map = x[:, 1:]

                            # reshape feature map: (B, num_features, num_patches, num_patches)
                            return x[:, 0], x_map.permute(0, 2, 1).view(x_map.shape[0], x_map.shape[-1], num_patches_h, num_patches_w)
                        mod_forward = forward
                    elif not return_feature_vector:
                        def forward(x):
                            self_ = self.model
                            B, nc, h, w = x.shape
                            num_patches_w = w // self_.patch_size
                            num_patches_h = h // self_.patch_size

                            # Reshape and permute the input tensor
                            x = self_._process_input(x)
                            n = x.shape[0]

                            # Expand the class token to the full batch
                            batch_class_token = self_.class_token.expand(n, -1, -1)
                            x = torch.cat([batch_class_token, x], dim=1)

                            x = self_.encoder(x)

                            # ignore classifier "token"
                            x = x[:, 1:]

                            # reshape feature map: (B, num_features, num_patches, num_patches)
                            return x.permute(0, 2, 1).view(x.shape[0], x.shape[-1], num_patches_h, num_patches_w)

                        mod_forward = forward
                    if mod_forward is not None:
                        self.model.forward = mod_forward
                        self.num_channels = self.model.hidden_dim

        if self.model is None:
            raise ValueError(
                f'Combination of weights {weights.name} and patch size {self.patch_size} not supported for {backbone_type}')

        if backbone_freezing is BackboneFreezing.completely:
            self.model.requires_grad_(False)
        elif backbone_freezing is BackboneFreezing.nothing:
            self.model.requires_grad_(True)
        elif backbone_freezing is BackboneFreezing.block:
            if weights is not PretrainedWeights.DINO:
                raise NotImplementedError(
                    f'Backbone freezing {backbone_freezing.name} not implemented for {weights.name} weights')
            if backbone_type is BackboneTypes.ViT_B_8 or backbone_type is BackboneTypes.ViT_B_16:
                self.model.patch_embed.requires_grad_(False)
                self.model.cls_token.requires_grad_(False)
                for i in range(len(self.model.blocks)):
                    self.model.blocks[i].requires_grad_(i >= num_frozen_blocks)
                self.model.norm.requires_grad_(True)

            else:
                LOGGER.info(f'Using backbone {backbone_type.name} with freezing method {backbone_freezing.name}. Using the bottleneck blocks after maxpool as blocks.')

                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)

                self.model.layer1.requires_grad_(0 >= num_frozen_blocks)
                self.model.layer2.requires_grad_(1 >= num_frozen_blocks)
                self.model.layer3.requires_grad_(2 >= num_frozen_blocks)
                self.model.layer4.requires_grad_(3 >= num_frozen_blocks)

        else:
            raise ValueError(f'Backbone freezing {backbone_freezing.name} unknown')

    def forward(self, *args):
        return self.model(*args)

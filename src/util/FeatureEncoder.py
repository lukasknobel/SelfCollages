import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import transforms
from .misc import get_resize_and_cropping_transforms
from ..models.ModelEnums import PretrainedWeights, BackboneTypes
from ..models.backbones import Backbone, BackboneFreezing

LOGGER = logging.getLogger()


class FeatureEncoder:

    ENCODED_FEATURES_FILE_NAME = 'encoded_features.pt'

    def __init__(self,  weights_dir, target_img_size, device, batch_size, num_workers, disable_tqdm):
        self.target_img_size = target_img_size
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.disable_tqdm = disable_tqdm

        self.encoder = Backbone(BackboneTypes.ViT_B_16, PretrainedWeights.DINO, weights_dir,
                                return_feature_vector=True, return_cls_self_attention=False,
                                backbone_freezing=BackboneFreezing.completely)

    @torch.no_grad()
    def encode_dataset(self, dataset, target_dir):
        LOGGER.debug(f'Encoding {len(dataset)} {dataset.__class__.__name__} images')

        encoded_features_path = os.path.join(target_dir, self.ENCODED_FEATURES_FILE_NAME)
        if os.path.isfile(encoded_features_path):
            LOGGER.info(f'Using encoded features saved in {encoded_features_path}')
            with open(encoded_features_path, 'rb') as f:
                encoded_features = torch.load(f, map_location='cpu')
            return encoded_features

        # backup dataset transform, which is modified during box prediction
        dataset_trans_backup = dataset.transform

        trans, resize_img_size = get_resize_and_cropping_transforms(self.target_img_size, cropping=False)
        if self.encoder.normalise_transform is not None:
            trans += [self.encoder.normalise_transform]
        dataset.transform = transforms.Compose(trans)

        encoded_features = torch.zeros((len(dataset), self.encoder.num_features), device=self.device)

        self.encoder = self.encoder.to(self.device)

        img_data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        for batch_idx, (imgs, _) in enumerate(tqdm(img_data_loader, leave=False, desc=f'Encoding {dataset.__class__.__name__}', disable=self.disable_tqdm)):

            if self.disable_tqdm and batch_idx % 100 == 0:
                LOGGER.debug(f'Encoding batch {batch_idx + 1}/{len(img_data_loader)}')

            batch_offset = batch_idx * self.batch_size
            imgs = imgs.to(self.device, non_blocking=True)

            batch_encoded_features = self.encoder(imgs)

            encoded_features[batch_offset:batch_offset + imgs.shape[0]] = batch_encoded_features

        with open(encoded_features_path, 'wb') as f:
            torch.save(encoded_features, f)
        LOGGER.debug(f'Saved encoded features in {encoded_features_path}')

        encoded_features = encoded_features.to('cpu')

        dataset.transform = dataset_trans_backup
        return encoded_features

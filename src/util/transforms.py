"""
Based on https://github.com/facebookresearch/detr/blob/main/datasets/transforms.py
and
https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
"""
import logging
import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch import nn

from . import Constants

LOGGER = logging.getLogger()


DENSITY_KEYS = [Constants.TARGET_DICT_DENSITY_MAPS, Constants.TARGET_DICT_ALL_DENSITY_MAPS]
MAP_KEYS = [Constants.TARGET_DICT_SEGMENTATIONS] + DENSITY_KEYS


def crop(image, region, target=None):
    cropped_image = F.crop(image, *region)

    if target is None:
        return cropped_image

    if Constants.TARGET_DICT_BOXES in target:
        boxes = target[Constants.TARGET_DICT_BOXES]
        squeeze = False
        if len(boxes.shape) == 1:
            boxes = boxes.unsqueeze(0)
            squeeze = True
        elif Constants.TARGET_DICT_NUM_ELEMENTS in target:
            boxes = boxes[:target[Constants.TARGET_DICT_NUM_ELEMENTS]]
        cropped_boxes = boxes.clone()
        i, j, h, w = region

        # get absolute box coordinates
        cropped_boxes[:, [0, 2]] *= image.shape[-1]
        cropped_boxes[:, [1, 3]] *= image.shape[-2]

        # compute new box
        cropped_boxes = cropped_boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), torch.tensor([h, w]))
        cropped_boxes = cropped_boxes.clamp(min=0)
        # area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        # remove elements for which the boxes or masks that have zero area
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        if not keep.all():
            LOGGER.warning(f'Cropped image does not contain bounding box')

        cropped_boxes = cropped_boxes.reshape(-1, 4)

        # normalise cropped boxes to get relative coordinates
        cropped_boxes[:, [0, 2]] /= w
        cropped_boxes[:, [1, 3]] /= h

        if squeeze:
            cropped_boxes = cropped_boxes[0]
            target[Constants.TARGET_DICT_BOXES] = cropped_boxes
        else:
            if Constants.TARGET_DICT_NUM_ELEMENTS in target:
                target[Constants.TARGET_DICT_BOXES][:target[Constants.TARGET_DICT_NUM_ELEMENTS]] = cropped_boxes
            else:
                target[Constants.TARGET_DICT_BOXES] = cropped_boxes

    for map_key in MAP_KEYS:
        if map_key in target:
            cropped_map = _crop_tensor(target[map_key], region)
            if not cropped_map.any():
                LOGGER.warning(f'Cropped image does not contain {map_key}')
            target[map_key] = cropped_map

    return cropped_image, target


def _crop_tensor(map_, region):
    i, j, h, w = region
    cropped_map = map_.clone()
    cropped_map = cropped_map[i:i + h, j:j + w]
    return cropped_map


def hflip(image, target=None):
    flipped_image = F.hflip(image)

    if target is None:
        return flipped_image

    if Constants.TARGET_DICT_BOXES in target:
        boxes = target[Constants.TARGET_DICT_BOXES]
        if len(boxes.shape) == 1:
            flipped_boxes = boxes[[2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([1, 0, 1, 0])
            target[Constants.TARGET_DICT_BOXES] = flipped_boxes
        else:
            flipped_boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([1, 0, 1, 0])
            if Constants.TARGET_DICT_NUM_ELEMENTS in target:
                target[Constants.TARGET_DICT_BOXES][:target[Constants.TARGET_DICT_NUM_ELEMENTS]] = flipped_boxes[:target[Constants.TARGET_DICT_NUM_ELEMENTS]]
            else:
                target[Constants.TARGET_DICT_BOXES] = flipped_boxes

    for map_key in MAP_KEYS:
        if map_key in target:
            flipped_map = _flip_tensor(target[map_key])
            target[map_key] = flipped_map

    return flipped_image, target


def _flip_tensor(map_):
    flipped_map = map_.clone()
    flipped_map = F.hflip(flipped_map)
    return flipped_map


def _resize_tensor(map_, size, max_size=None, binary=False, keep_sum=False):
    resized_map = map_.clone()
    if binary:
        interpolation = F.InterpolationMode.NEAREST
    else:
        interpolation = F.InterpolationMode.BILINEAR
    resized_map = F.resize(resized_map.unsqueeze(0).unsqueeze(0), size, interpolation, max_size).squeeze(0).squeeze(0)
    if keep_sum:
        resized_map *= map_.sum() / resized_map.sum()
    return resized_map


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target=None):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, region, target)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target=None):
        if random.random() < self.p:
            return hflip(img, target)
        if target is None:
            return img
        return img, target


class ToTensor(object):
    def __call__(self, img, target=None):
        if target is None:
            return F.to_tensor(img)
        return F.to_tensor(img), target


class Resize(nn.Module):
    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR, max_size=None):
        super().__init__()
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size

    def __call__(self, img, target=None):
        if target is None:
            return F.resize(img, self.size, self.interpolation, self.max_size)
        for map_key in MAP_KEYS:
            if map_key in target:
                binary = map_key is Constants.TARGET_DICT_SEGMENTATIONS
                keep_sum = map_key in DENSITY_KEYS
                target[map_key] = _resize_tensor(target[map_key], self.size, max_size=self.max_size, binary=binary, keep_sum=keep_sum)

        return F.resize(img, self.size, self.interpolation, self.max_size), target

    def __repr__(self) -> str:
        detail = f"(size={self.size}, interpolation={self.interpolation.value})"
        return f"{self.__class__.__name__}{detail}"


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target

    def reverse(self, norm_image, target=None):
        image = norm_image.clone()
        image *= torch.tensor(self.std, device=image.device).view(3, 1, 1)
        image += torch.tensor(self.mean, device=image.device).view(3, 1, 1)
        if target is None:
            return image
        return image, target


class ColorJitter:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = T.ColorJitter(*args, **kwargs)

    def __call__(self, image, target=None):
        image = self.transform(image)
        if target is None:
            return image
        return image, target


class RandomApply:

    def __init__(self, transform, p=0.5):
        super().__init__()

        self.transform = transform
        self.p = p

    def forward(self, img):
        if self.p < torch.rand(1):
            return img
        img = self.transform(img)
        return img

    def __call__(self, image, target=None):
        if torch.rand(1) <= self.p:
            image, target = self.transform(image, target)
        if target is None:
            return image
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        if target is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, target = t(image, target)
            return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


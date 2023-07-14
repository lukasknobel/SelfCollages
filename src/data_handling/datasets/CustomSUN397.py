import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import PIL.Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

from .MSODataset import MSODataset
from ..DatasetSplits import DatasetSplits
from ...util import Constants
from ...util.SSLBoxAnnotator import SSLBoxAnnotator

LOGGER = logging.getLogger()


class SUN397(VisionDataset):
    """This is largely a copy of the PyTorch SUN397 class without the hardcoded "SUN397" subdir, class_name_path argument

    """
    """`The SUN397 Data Set <https://vision.princeton.edu/projects/2010/SUN/>`_.

    The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of
    397 categories with 108'754 images.

    Args:
        root (string): Root directory of the dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _DATASET_URL = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    _DATASET_MD5 = "8ca2778205c41d23104230ba66911c7a"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        sub_dir=None,
        class_name_path=''
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        if sub_dir is None:
            self._data_dir = Path(self.root)
        else:
            self._data_dir = Path(self.root) / sub_dir

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if class_name_path == '':
            class_name_path = os.path.join(self._data_dir, 'ClassName.txt')

        with open(class_name_path) as f:
            self.classes = [c[3:].strip() for c in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._image_files = list(self._data_dir.rglob("sun_*.jpg"))

        self._labels = [
            self.class_to_idx["/".join(path.relative_to(self._data_dir).parts[1:-1])] for path in self._image_files
        ]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _check_exists(self) -> bool:
        return self._data_dir.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._DATASET_URL, download_root=self.root, md5=self._DATASET_MD5)


class CustomSUN397(SUN397):

    SPLIT_META_DIR = 'Partitions'
    TEST_SPLIT_FILE_NAME = 'Testing_01.txt'
    SOS_TEST_SUN_IMG_FILE_NAME = 'mso_sun_imgs.txt'
    SEPARATOR = ','

    def __init__(self, processed_base_dir, *args, split=DatasetSplits.TRAIN, box_annotator: Optional[SSLBoxAnnotator] = None,
                 img_dir='', **kwargs):
        self.processed_dir = os.path.join(processed_base_dir, 'SUN397')
        if img_dir == '':
            # if img_dir is not specified, we assume that the data is in a subdirectory of the processed_dir
            img_dir = self.processed_dir
            sub_dir = 'SUN397'
        else:
            # if the img_dir is specified, it points directly to the image directory, no sub_dir is needed
            sub_dir = None
        super().__init__(img_dir, *args, sub_dir=sub_dir, **kwargs)

        self.split = split
        meta_path = os.path.join(self.processed_dir, CustomSUN397.SPLIT_META_DIR)

        split_file = os.path.join(meta_path, CustomSUN397.TEST_SPLIT_FILE_NAME)

        with open(split_file, 'r') as f:
            test_sample_paths = f.readlines()
        test_sample_paths = [self._data_dir.joinpath(p[1:-1]) for p in test_sample_paths]

        name_idx_mapping = {file: idx for idx, file in enumerate(self._image_files)}
        s = set(self._image_files)
        t = set(test_sample_paths)
        if self.split is DatasetSplits.TRAIN:
            split_img_files = list(s.difference(t))
        elif self.split is DatasetSplits.TEST:
            split_img_files = list(s.intersection(t))
        else:
            raise ValueError(f'Split {split.name} is not supported')
        # remove SUN samples from train/test split that are used in the SOS test/train split
        split_img_files = self._remove_mso_sun_imgs(processed_base_dir, split_img_files)
        split_img_files = sorted(split_img_files)
        split_labels = [self._labels[name_idx_mapping[name]] for name in split_img_files]
        self._image_files = split_img_files
        self._labels = split_labels

        self.box_annotator = box_annotator
        self.boxes = None
        if self.box_annotator is not None:
            self.boxes = self.box_annotator.annotate_boxes(self, self.processed_dir)

    def __getitem__(self, idx):
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")
        target = {Constants.TARGET_DICT_CLASSES: label}
        if self.boxes is not None:
            target[Constants.TARGET_DICT_BOXES] = self.boxes[idx]

        if self.transform:
            image, target = self.transform(image, target)

        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def _remove_mso_sun_imgs(self, base_dir, split_img_files):
        if self.split is DatasetSplits.TRAIN:
            mso_sun_img_file_path = os.path.join(self.processed_dir, CustomSUN397.SOS_TEST_SUN_IMG_FILE_NAME)
            if os.path.isfile(mso_sun_img_file_path):
                with open(mso_sun_img_file_path, 'r') as f:
                    mso_split_sun_images = f.read()
                    mso_split_sun_images = mso_split_sun_images.split(CustomSUN397.SEPARATOR)
            else:
                mso_split_sun_images = [n[4:] for n in MSODataset(base_dir, in_memory=False).file_names if
                                        'sun' in n.lower()]
                with open(mso_sun_img_file_path, 'w') as f:
                    f.write(CustomSUN397.SEPARATOR.join(mso_split_sun_images))
            old_len = len(split_img_files)
            split_img_files = [f for f in split_img_files if f.name not in mso_split_sun_images]
            LOGGER.debug(
                f'Ignore {old_len - len(split_img_files)} images in {self.split.name} partition that are used in MSO')
        return split_img_files

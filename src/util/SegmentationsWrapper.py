import logging
import os

import torch
import torchvision.transforms.functional as F

from . import transforms
from .misc_enums import SegmentationTypes

LOGGER = logging.getLogger()


class SegmentationsWrapper:

    segmentations_subdir = 'segmentations'

    def __init__(self, dataset_dir: str, output_dir: str, dataset_size,
                 segmentation_type: SegmentationTypes = SegmentationTypes.selfmask,
                 subset_idxs=None, assume_sorted_subidxs=False, in_memory: bool = False,
                 down_sampling_limit=330000, downsampling_size=112):
        self.dataset_dir = dataset_dir
        self.assume_sorted_subidxs = assume_sorted_subidxs
        if subset_idxs is not None:
            self.subset_idxs = torch.tensor(subset_idxs)
            if self.assume_sorted_subidxs:
                self.subset_idxs = self.subset_idxs.sort()[0]
            self.dataset_size = len(self.subset_idxs)
        else:
            self.subset_idxs = None
            self.dataset_size = dataset_size

        if self.dataset_size > down_sampling_limit:
            self.segmentation_resize = transforms.Resize((downsampling_size, downsampling_size), interpolation=F.InterpolationMode.NEAREST)
        else:
            self.segmentation_resize = None

        self.segmentation_type = segmentation_type
        self.in_memory = in_memory
        self.segmentation_dir = os.path.join(output_dir, self.segmentations_subdir, self.segmentation_type.name)
        self._setup()

    def _setup(self):
        self.segmentation_files = sorted([os.path.join(self.segmentation_dir, f) for f in os.listdir(self.segmentation_dir)])
        self.index_to_filename_and_offset = None
        self.seg_file = None

        if self.assume_sorted_subidxs:
            cur_seg_file_idx = 0

        if self.in_memory:
            LOGGER.debug(f'Loading {self.dataset_size} segmentations into memory')
            self.seg_file = None
            orig_idx_2_subset_idx = None
            if self.subset_idxs is not None:
                orig_idx_2_subset_idx = {orig_idx.item(): subset_idx for subset_idx, orig_idx in enumerate(self.subset_idxs)}

            seg_file_sorted_by_first_idx = sorted(
                [(int(seg_file.split('.')[-2].rsplit('_', 2)[-2]), int(seg_file.split('.')[-2].rsplit('_', 1)[-1]), seg_file) for seg_file in self.segmentation_files],
                key=lambda x: x[0]
            )
            for seg_file_orig_start_idx, seg_file_orig_end_idx, seg_file in seg_file_sorted_by_first_idx:
                seg_map = torch.load(seg_file)
                if self.segmentation_resize is not None:
                    seg_map = self.segmentation_resize(seg_map)
                if self.seg_file is None:
                    self.seg_file = torch.zeros((self.dataset_size, *seg_map.shape[1:]), dtype=torch.bool)
                if self.subset_idxs is None:
                    self.seg_file[seg_file_orig_start_idx:seg_file_orig_end_idx+1] = seg_map
                else:
                    # only pick the segmentations that are in the subset
                    if self.assume_sorted_subidxs:
                        # get indices relative to the start of the current segmentation file
                        relative_orig_idxs = self.subset_idxs - seg_file_orig_start_idx
                        # only keep indices that are in the current segmentation file
                        relative_orig_idxs = relative_orig_idxs[(relative_orig_idxs > 0) & (relative_orig_idxs <= (seg_file_orig_end_idx - seg_file_orig_start_idx))]
                        self.seg_file[cur_seg_file_idx:cur_seg_file_idx+len(relative_orig_idxs)] = seg_map[relative_orig_idxs]
                        cur_seg_file_idx += len(relative_orig_idxs)
                    else:
                        for i in range(len(seg_map)):
                            orig_idx = i + seg_file_orig_start_idx
                            if orig_idx in orig_idx_2_subset_idx:
                                self.seg_file[orig_idx_2_subset_idx[orig_idx]] = seg_map[i]
            if self.seg_file.shape[0] != self.dataset_size:
                raise ValueError(f'Number of segmentations ({self.seg_file.shape[0]}) does not match dataset size ({self.dataset_size})')
        else:
            self.index_to_filename_and_offset = []
            for seg_file_idx, seg_file in enumerate(self.segmentation_files):
                seg_start, seg_end = seg_file.split('.')[-2].split('_')[-2:]
                seg_start = int(seg_start)
                seg_end = int(seg_end) + 1
                for offset, idx in enumerate(range(seg_start, seg_end)):
                    self.index_to_filename_and_offset.append((idx, seg_file_idx, offset))

            self.index_to_filename_and_offset = sorted(self.index_to_filename_and_offset, key=lambda x: x[0])
            if self.subset_idxs is not None:
                self.index_to_filename_and_offset = [self.index_to_filename_and_offset[idx] for idx in self.subset_idxs]
            if len(self.index_to_filename_and_offset) != self.dataset_size:
                raise ValueError(f'Number of segmentations ({len(self.index_to_filename_and_offset)}) does not match dataset size ({self.dataset_size})')

    def get_segmentation(self, index: int, width: int, height: int):
        if self.in_memory:
            segmentation = self.seg_file[index]
        else:
            _, seg_filename_idx, offset = self.index_to_filename_and_offset[index]
            seg_filename = self.segmentation_files[seg_filename_idx]
            segmentations = torch.load(seg_filename)
            segmentation = segmentations[offset]

        if width != segmentation.shape[1] or height != segmentation.shape[0]:
            segmentation = torch.nn.functional.interpolate(segmentation.unsqueeze(0).unsqueeze(0).to(torch.float), size=(height, width), mode='nearest').squeeze(0).squeeze(0).to(torch.bool)

        return segmentation

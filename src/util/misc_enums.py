from enum import Enum


class AnnotationTypes(Enum):
    NO = 0
    DINO_SEG_16 = 1
    DINO_SEG_8 = 2
    SPECTRAL_DINO_SEG_16 = 3
    SPECTRAL_DINO_SEG_8 = 4


class SegmentationTypes(Enum):
    selfmask = 0


"""
    NoiseDataset configurations
"""
class NoiseTypes(Enum):
    StyleGAN_Oriented = 0


"""
    CustomImageNet configurations
"""
class ClusterFilter(Enum):
    No = 0
    Top10 = 1
    AtLeast150 = 2
    Top20 = 3
    Top50 = 4
    Top100 = 5
    Top200 = 6
    Top500 = 7
    Top2 = 8

"""
    PatchDataset configurations
"""
class ConstructionModes(Enum):
    Pasting = 0
    Segmentations = 1
    Synthetic_shapes = 2


class SyntheticShapes(Enum):
    Square = 0
    Circle = 1
    Triangle = 2


class SyntheticColours(Enum):
    Red = 0
    Green = 1
    Blue = 2
    Yellow = 3


class ClusterSimilarity(Enum):
    No = 0
    Top10 = 1
    Top100 = 2
    Between10_100 = 3
    Between100_1000 = 4
    Bottom1000 = 5
    Between10_100_C = 6
    Between10_1000_C = 7
    Between100_1000_C = 8


class Blending(Enum):
    No = 0


class DensityMapTypes(Enum):
    Segmentation = 0
    BoxCenters = 1
    BlurredSegmentation = 2


class LabelDistributions(Enum):
    at_least_3 = 2


class BaseDatasets(Enum):
    ImageNet_SUN = 0
    ImageNet_StyleGAN_Noise = 1
    ImageNet = 2

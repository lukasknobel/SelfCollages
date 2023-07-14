import os

from torchvision.datasets import ImageFolder

from ...util.misc_enums import NoiseTypes


class NoiseDataset(ImageFolder):
    def __init__(self, base_dir, method: NoiseTypes = NoiseTypes.StyleGAN_Oriented, large_scale=True, **kwargs):

        root = os.path.join(base_dir, 'noise_dataset')
        if large_scale:
            root = os.path.join(root, 'large_scale')
        if method is NoiseTypes.StyleGAN_Oriented:
            root = os.path.join(root, 'stylegan-oriented')

        super().__init__(root, **kwargs)

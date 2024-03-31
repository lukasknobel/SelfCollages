import enum


class ModelTypes(enum.Enum):
    UnCounTR = 0


class BackboneTypes(enum.Enum):
    ViT_B_8 = 0
    ViT_B_16 = 1
    ViT_S_16 = 2
    ViT_S_14 = 3
    ViT_B_14 = 4
    ViT_L_14 = 5
    ViT_G_14 = 6


class Heads(enum.Enum):
    LINEAR = 0
    MLP = 1


class PretrainedWeights(enum.Enum):
    RANDOM = 0
    DINO = 1
    IMAGENET = 2
    LEOPART = 3
    DINOv2 = 4

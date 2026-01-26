from segmentator.nnunetv2_custom.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import (
    nnUNetTrainerDA5,
)
from segmentator.nnunetv2_custom.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainer_onlyMirror01,
)


class nnUNetTrainer_onlyMirror01_DA5(nnUNetTrainer_onlyMirror01, nnUNetTrainerDA5):
    pass

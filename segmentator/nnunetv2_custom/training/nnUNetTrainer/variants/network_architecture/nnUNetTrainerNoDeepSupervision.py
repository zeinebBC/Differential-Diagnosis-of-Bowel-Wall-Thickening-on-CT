import torch

from segmentator.nnunetv2_custom.training.nnUNetTrainer.nnUNetTrainer import (
    nnUNetTrainer,
)


class nnUNetTrainerNoDeepSupervision(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False

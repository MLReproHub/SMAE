import timeit
from typing import Optional

import torch
from torch import nn


class PatchWise(nn.Module):
    def __init__(self, criterion: nn.Module):
        super(PatchWise, self).__init__()
        self.criterion = criterion

    def forward(self, patches, original_patches, mask_indices=None):
        return self.criterion(
            input=patches,
            target=original_patches
        ) if mask_indices is None else self.criterion(
            input=torch.gather(patches, dim=1, index=mask_indices[:, :, :patches.shape[2]]),
            target=torch.gather(original_patches, dim=1, index=mask_indices[:, :, :patches.shape[2]])
        )

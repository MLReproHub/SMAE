from typing import Tuple

import torch
from torch import nn, tensor, Tensor

from loss.patch_wise import PatchWise
from loss.perceptual import SqueezePerceptual


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, pixel_criterion: nn.Module, perceptual_criterion: nn.Module, device='cuda'):
        super().__init__()
        self.pixel_loss = PatchWise(criterion=pixel_criterion)
        self.perceptual_loss = SqueezePerceptual(criterion=perceptual_criterion)
        self.log_var1 = nn.Parameter(tensor(-1.0), requires_grad=True)
        self.log_var2 = nn.Parameter(tensor(-6.0), requires_grad=True)

    @property
    def w_pixel(self) -> float:
        return torch.exp(-self.log_var1.detach()).item()

    @property
    def w_perceptual(self) -> float:
        return torch.exp(-self.log_var2.detach()).item()

    def forward(self, x_hat, x, patches_hat, patches) -> Tuple[Tensor, Tensor, Tensor]:
        # Unreduced loss
        pixel_loss = self.pixel_loss(patches_hat, patches).view(patches_hat.shape[0], -1).mean(-1, keepdims=True)
        perceptual_loss = self.perceptual_loss(x_hat, x).view(x_hat.shape[0], -1).mean(-1, keepdims=True)

        precision1 = torch.exp(-self.log_var1)
        precision2 = torch.exp(-self.log_var2)
        loss = torch.sum(precision1 * pixel_loss + self.log_var1, dim=-1)
        loss = loss + torch.sum(precision2 * perceptual_loss + self.log_var2, dim=-1)
        uncertainty_weighted_loss = loss.mean()

        return uncertainty_weighted_loss, pixel_loss.detach().mean(), perceptual_loss.detach().mean()

    def state_dict(self, *args, **kwargs) -> dict:
        return {'log_var1': self.log_var1, 'log_var2': self.log_var2}

    def load_state_dict(self, sd: dict, *args) -> None:
        self.log_var1 = sd['log_var1']
        self.log_var2 = sd['log_var2']

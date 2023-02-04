from functools import partial
from typing import Callable
from unittest import TestCase

import torch
from einops import rearrange, reduce, repeat
from torch import nn

from model.layer import SwapAxes
from model.mae import Lambda


class TestMAE(TestCase):
    def setUp(self) -> None:
        self.x = torch.zeros(1, 3, 224, 224)
        self.x[:, 0] = 1.0
        self.x[:, 1] = 2.0
        self.x[:, 2] = 2.0

    def patchify(self, x, patch_size: int):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        assert x.shape[2] == x.shape[3] and x.shape[2] % patch_size == 0
        h = w = x.shape[2] // patch_size
        return rearrange(x, 'n c (h p) (w q) -> n (p q c) (h w)', p=patch_size, h=h, w=w)

    def test_unfold_shape(self):
        ks, d_model = 8, 256
        project = nn.Linear(3 * ks ** 2, d_model)
        #   - unfold + project
        unfold = nn.Unfold(kernel_size=ks, stride=ks)
        unfold_swap = nn.Sequential(unfold, SwapAxes(1, 2))
        unfold_project = nn.Sequential(unfold_swap, project)
        #   - conv version
        conv = nn.Sequential(
            nn.Conv2d(3, 3 * ks ** 2, kernel_size=ks, stride=ks, bias=False),
            nn.Flatten(2)
        )
        conv[0].weight.data.fill_(1 / (5 * ks ** 2))
        conv_project = nn.Sequential(
            nn.Conv2d(3, d_model, kernel_size=ks, stride=ks, bias=False),
            nn.Flatten(2),
            SwapAxes(1, 2),
        )

        # Test
        patches_uf = unfold(self.x.clone())
        patches_pf = self.patchify(self.x.clone(), ks)
        patches_cv = conv(self.x.clone())
        self.assertListEqual(list(patches_uf.shape), list(patches_pf.shape))
        self.assertListEqual(list(patches_uf.shape), list(patches_cv.shape))
        self.assertListEqual(list(conv_project(self.x.clone()).shape), list(unfold_project(self.x.clone()).shape))
        # This fails
        # self.assertTrue(torch.allclose(patches_uf, patches_pf, rtol=1e-3, atol=1e-3))

"""
A Feature Pyramid Network for use in object detection and instance segmentation.
[Lin et al. 2017](https://doi.org/10.48550/arXiv.1612.03144).
The ViT adaption follows that of [Li et al. 2021](https://doi.org/10.48550/arXiv.2111.11429)
"""
from collections import OrderedDict
from math import log2, sqrt
from typing import Dict, List, Optional, Callable

from torch import nn, Tensor
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, LastLevelMaxPool, FeaturePyramidNetwork


class ViTBackboneWithFPN(nn.Module):
    """
    Custom adaptation of official torchvision code.

    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Args:
        bottom_up_encoder (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(
            self,
            bottom_up_encoder: nn.Module,
            return_layers: Dict[str, str],
            in_channels_list: List[int],
            out_channels: int,
            extra_blocks: Optional[ExtraFPNBlock] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(bottom_up_encoder.backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.multi_scale_adaptor = MultiScaleAdaptor(patch_size=bottom_up_encoder.kernel_size,
                                                     d_model=bottom_up_encoder.d_model)
        self.out_channels = out_channels
        self.bottom_up_encoder = bottom_up_encoder

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x, _ = self.bottom_up_encoder.tokenize(x)
        # TODO we get away without appending dummy class token here? Would simplify things. The only thing holding us
        #  back might be the fact that the encoder is "used" to the class token.
        x = self.body(x)
        x = self.multi_scale_adaptor(x)
        x = self.fpn(x)
        return x


class MultiScaleAdaptor(nn.Module):
    """
    Scales the transformer representations to work with FPN.
    Own implementation as described in [Li et al. 2021](https://doi.org/10.48550/arXiv.2111.11429)
    """
    # The patch size scales to which the interim ViT feature maps are mapped
    STRIDES = [4, 8, 16, 32]

    def __init__(self, patch_size, d_model):
        super(MultiScaleAdaptor, self).__init__()
        # patch_size * 2**scale = stride
        self.scales = [int(log2(s / patch_size)) for s in self.STRIDES]
        self.adaptor = self.build_adaptor(d_model)

    def build_adaptor(self, d_model):
        modules = OrderedDict()
        for i, scale in enumerate(self.scales):
            if scale < 0:
                modules[f'{i}_upsample'] = UpSampler(n=abs(scale), d_model=d_model)
            elif scale > 0:
                modules[f'{i}_downsample'] = DownSampler(n=scale)
            else:
                modules[f'{i}_identity'] = nn.Identity()
        return modules

    def forward(self, x):
        """
        Scales the input features of each intermediate layer in order of increasing stride.
        :param x: An OrderedDict holding the intermediate transformer features.
        :return: An OrderedDict holding the scaled feature maps for each intermediate layer.
        """
        # The scaled feature maps for each interim layer
        scaled_features = OrderedDict()
        for (_, module), (layer_name, features) in zip(self.adaptor.items(), x.items()):
            patches_per_axis = int(sqrt(features.shape[1]))
            # Reshape to 2d and reorder dimensions to traditional convolution dims. (B, C, H, W)
            features_2d = features.reshape(features.shape[0], patches_per_axis, patches_per_axis, features.shape[2]) \
                .permute(0, 3, 1, 2)

            scaled_features[layer_name] = module(features_2d)

        return scaled_features


class DownSampler(nn.Module):
    """
    Downsamples feature maps, preserving the number of feature maps.
    """

    def __init__(self, n):
        """
        :param n: How many times each dimension should be doubled. n >= 1
        """
        super(DownSampler, self).__init__()
        assert n >= 1, "n has to be greater or equal to one"
        self.pool = nn.MaxPool2d(kernel_size=2 * n, stride=2 * n)

    def forward(self, x):
        return self.pool(x)


class UpSampler(nn.Module):
    """
    Upsamples feature maps, preserving the number of feature maps.
    """

    def __init__(self, n, d_model):
        """
        :param n: How many times each dimension should be doubled. n >= 1
        """
        super(UpSampler, self).__init__()
        if n > 1:
            self.additional_upsample_blocks = nn.Sequential(*[self.upsample_block(d_model) for _ in range(n - 1)])
        self.final_upsample = nn.ConvTranspose2d(in_channels=d_model, out_channels=d_model, kernel_size=2, stride=2)

    def forward(self, x):
        if self.additional_upsample_blocks is not None:
            x = self.additional_upsample_blocks(x)
        return self.final_upsample(x)

    @staticmethod
    def upsample_block(d_model):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=d_model, out_channels=d_model, kernel_size=2, stride=2),
            nn.LayerNorm(normalized_shape=d_model),
            nn.GELU()
        )


def test_integration(bottom_up_encoder, dl, training_setup):
    # Li et al. take every fourth layer, but we only have 7 layers, soo...
    return_layers = [3, 4, 5, 6]
    fpn_dim = 256
    fpn = ViTBackboneWithFPN(bottom_up_encoder=bottom_up_encoder,
                             return_layers={str(v): str(v) for v in return_layers},
                             in_channels_list=[bottom_up_encoder.d_model for _ in return_layers],
                             out_channels=fpn_dim).to(training_setup.device)
    x, y = next(iter(dl))
    out = fpn(x)
    print()

    # In order to run, do:
    """
    bottom_up_enc = FPNBottomUpEncoder(model).to(training_setup.device)
    test_integration(bottom_up_enc, dl_train, training_setup)
    """

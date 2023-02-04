from functools import partial
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from model.layer import AppendClassToken, PositionalEncoding, AppendMaskTokens, Normalize, ReorderToBlockWiseMask, ShufflePatches, Lambda, \
    Unfold
from utilities.train import IPretrainer


class MAE(nn.Module, IPretrainer):
    def __init__(self, *,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 masking_ratio: float,
                 kernel_size: int,
                 normalize: bool,
                 masking_strategy: str = 'random-sampling',
                 which_unfolder: str = 'unfold'):
        super(MAE, self).__init__()
        encoder_dim = encoder.d_model
        decoder_dim = decoder.d_model

        self.trained = False

        self.masking_ratio = masking_ratio
        self.normalize = normalize

        self.encoder = encoder
        self.decoder = decoder

        self.kernel_size = kernel_size
        self.unfold = Unfold(patch_size=kernel_size, which_impl=which_unfolder)
        self.fold = Lambda(partial(F.fold, kernel_size=kernel_size, stride=kernel_size))
        self.patch_projection = nn.Linear(3 * kernel_size ** 2, encoder_dim)
        self.append_cls_token = AppendClassToken(d_model=encoder_dim)
        self.encoder_pos_enc = PositionalEncoding(d_model=encoder_dim, max_len=1000)
        self.decoder_pos_enc = PositionalEncoding(d_model=decoder_dim, max_len=1000)
        self.append_mask_tokens = AppendMaskTokens(decoder_dim)
        self.latent_projection = nn.Linear(encoder_dim, decoder_dim)
        self.output_projection = nn.Linear(decoder_dim, 3 * kernel_size ** 2)
        self.patch_wise_normalize = Normalize(dim=-1)

        if masking_strategy == 'random-sampling':
            self.reorder_tokens = ShufflePatches()
        elif masking_strategy == 'block-wise':
            self.reorder_tokens = ReorderToBlockWiseMask(masking_ratio)
        else:
            assert False, f"Unknown masking strategy setting: \"{masking_strategy}\""

    @property
    def d_model(self) -> int:
        return self.encoder.d_model

    def forward(self, x: torch.Tensor, blocks=None):
        tokens, original_patches = self.tokenize(x)

        # Randomly shuffle the tokens and remove some of them
        tokens, indices, mask_indices = self.generate_mask(tokens, blocks)

        z = self.encode(tokens)

        reconstruction, reconstructed_patches = self.decode(z, indices, n_drop=mask_indices.shape[1],
                                                            img_shape=x.shape[2:])

        return reconstruction, reconstructed_patches, original_patches, mask_indices

    def tokenize(self, x):
        patches = self.unfold(x).swapaxes(1, 2)  # BCHW -> BN(3*k*k)
        original_patches = patches.detach().clone()  # Save copy of original patches (to be used in loss calculations)
        if self.normalize:
            original_patches = self.patch_wise_normalize(original_patches)
        tokens = self.patch_projection(patches)  # BN(3*k*k) -> BD(3*k*k)
        tokens = self.encoder_pos_enc(tokens)
        return tokens, original_patches

    def generate_mask(self, tokens, blocks):
        tokens, indices = self.reorder_tokens(tokens, blocks)

        # Remove the last n_drop tokens, where n_drop depends on the masking ratio.
        n_drop = int(tokens.shape[1] * self.masking_ratio)
        tokens = tokens[:, :-n_drop]
        mask_indices = indices[:, -n_drop:]

        return tokens, indices, mask_indices

    def encode(self, tokens):
        # Append class token to encoder input sequence, as per He (2022)
        tokens = self.append_cls_token(tokens)

        z = self.encoder(tokens)
        return z

    def decode(self, z, indices, n_drop, img_shape: Tuple[int, int]):
        # Drop class token
        z_without_class_token = z[:, :-1, :]

        tokens = self.latent_projection(z_without_class_token)
        tokens = self.append_mask_tokens(tokens, n_dropped=n_drop)
        # Un-shuffle
        tokens.scatter_(dim=1, index=indices[:, :, :tokens.shape[2]], src=tokens.clone())
        tokens = self.decoder_pos_enc(tokens)
        tokens = self.decoder(tokens)
        reconstructed_patches = self.output_projection(tokens)

        # Fold reconstructed patches into original image dimensions
        reconstruction = self.fold(reconstructed_patches.swapaxes(1, 2), img_shape)
        return reconstruction, reconstructed_patches

    def freeze_encoder(self, freeze_n_layers: int or str or None = None):
        if freeze_n_layers is None:
            return

        #   - freeze projection
        for _, p in self.patch_projection.named_parameters():
            p.requires_grad = False
        self.patch_projection.eval()

        #   - freeze encoder layers
        self.encoder.freeze_layers(freeze_n_layers)


class ClassifierEncoder(nn.Module):
    def __init__(self, model, freeze_n_layers='all'):
        super(ClassifierEncoder, self).__init__()
        # TODO the model should be cloned if we reuse the same MAE model for different fine-tunings, like if we ablate
        #  the number of fine-tuned layers.
        self.model = model
        self.model.freeze_encoder(freeze_n_layers)

    def forward(self, x: torch.Tensor):
        tokens, _ = self.model.tokenize(x)

        z = self.model.encode(tokens)

        class_token_result = z[:, -1, :]
        return class_token_result

    @property
    def d_model(self):
        return self.model.d_model


class FPNBottomUpEncoder(nn.Module):
    def __init__(self, model):
        super(FPNBottomUpEncoder, self).__init__()
        self.model = model

    @property
    def backbone(self):
        return self.model.encoder.enc_layers

    @property
    def tokenize(self):
        return self.model.tokenize

    @property
    def kernel_size(self):
        return self.model.kernel_size

    @property
    def d_model(self):
        return self.model.d_model


if __name__ == '__main__':
    from utilities.config import ConfigReader

    mae_ = ConfigReader.load_all(model_key='mae', model_config='debug')[0]
    x_ = torch.rand(1, 3, 224, 224)
    x_hat_, x_hat_patches_, x_patches_, mask_ = mae_(x_)
    print(x_hat_.shape, x_hat_patches_.shape)

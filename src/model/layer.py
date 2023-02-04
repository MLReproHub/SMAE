from typing import Tuple, Callable
import math

import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor


class AppendMaskTokens(nn.Module):
    """
    AppendMaskTokens Class:
    Appends n number of mask tokens to the end of a sequence.
    """

    def __init__(self, d_model: int):
        """
        AppendMaskTokens class constructor.
        :param int d_model: model dimensionality
        """
        super(AppendMaskTokens, self).__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        nn.init.normal_(self.mask_token, std=1e-3)

    def forward(self, tokens: Tensor, n_dropped: int) -> Tensor:
        """
        :param Tensor tokens: tensor of shape (batch_size, seq_len, embedding_dim)
        :param int n_dropped: number of (mask) tokens to append at the end of the sequence (i.e. how many tokens were
                              dropped during random sampling)
        :return: x with n mask tokens appended (batch_size, seq_len+n, embedding_dim)
        """
        mask_tokens = self.mask_token.expand(tokens.shape[0], n_dropped, -1)
        return torch.cat((tokens, mask_tokens), dim=1)


class Lambda(nn.Module):
    """
    Lambda Class:
    An easy way to create a pytorch layer for a simple `func`.
    """

    def __init__(self, func: Callable):
        """
        Lambda class constructor.
        Create a layer that simply calls `func` with `x`.
        """
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class Normalize(nn.Module):
    """
    Normalizes the values patch-wise with respect to first and second moment.
    """

    def __init__(self, dim):
        super(Normalize, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: tensor of shape (batch_size, seq_length, embedding_dim)
        :return: x, but every patch in the sequence is normalized w.r.t first and second moment.
        """
        eps = torch.ones_like(x) * torch.finfo(torch.float32).eps
        x = x - x.mean(dim=self.dim, keepdim=True)
        x = x / torch.maximum(x.std(dim=self.dim, unbiased=False, keepdim=True), eps)
        return x


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding Class:
    1D positional sequence encoding based on sine and cosine waves.
    Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Source for encoding: "Attention is all you need", Vaswani et al.
    """

    def __init__(self, d_model: int, max_len: int, dropout_p: float or None = None):
        super().__init__()
        # 1) Create sinusoidal encoding
        # Create position vector and reshape to column vector.
        pos = torch.arange(max_len).unsqueeze(1)
        denominator = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        # Encoding for each input in sequence should be same dimension as input, so that they may be added.
        encoding = torch.zeros(1, max_len, d_model)
        # Set even dimensions to sin function
        encoding[0, :, 0::2] = torch.sin(pos / denominator)
        # Set odd dimensions to cos function
        encoding[0, :, 1::2] = torch.cos(pos / denominator)
        # Save as non-trainable parameters
        self.register_buffer('encoding', encoding)

        # 2) Create dropout layer
        self.use_dropout = False
        if dropout_p is not None:
            self.dropout = nn.Dropout(p=dropout_p)
            self.use_dropout = True

    def forward(self, x: Tensor) -> Tensor:
        """
        :param Tensor x: shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.encoding[0, :x.shape[1]]
        if self.use_dropout:
            return self.dropout(x)
        return x

    def plot(self):
        fig, ax = plt.subplots()
        # len x dim
        ax.imshow(self.encoding[0, :, :])
        ax.set_xlabel('dim')
        ax.set_ylabel('seq')
        plt.show()


class PrependClassToken(nn.Module):
    """
    PrependClassToken Class:
    Module to prepend a learnable class token to every sequence in batch.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # The class token does not carry any information in itself. The hidden state corresponding to this token at the
        # end of the transformer will be inferred by all other tokens in the sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: dimensions [batch_size, seq_len, embedding_dim]
        :return: x prepended with class token [batch_size, seq_len+1, embedding_dim]
        """
        return torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)


class AppendClassToken(nn.Module):
    """
    AppendClassToken Class:
    Module to append a learnable class token to every sequence in batch.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # The class token does not carry any information in itself. The hidden state corresponding to this token at the
        # end of the transformer will be inferred by all other tokens in the sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: dimensions [batch_size, seq_len, embedding_dim]
        :return: x appended with class token [batch_size, seq_len+1, embedding_dim]
        """
        return torch.cat((x, self.cls_token.expand(x.shape[0], -1, -1)), dim=1)


class ReorderToBlockWiseMask(nn.Module):
    """
    Reorders the input along the second dimension independently for each
    item in the batch such that the first masking_ratio proportion of items
    constitute the items to keep after a block-wise masking operation
    """

    def __init__(self, masking_ratio):
        super(ReorderToBlockWiseMask, self).__init__()
        self.masking_ratio = masking_ratio

    def generate_mask_block(self, input_height, input_width, min_size, max_size):
        # Generate random numbers needed for this block
        random_numbers = torch.rand([4, ])

        # s
        block_size_range = max_size - min_size
        block_size = min_size + random_numbers[0] * block_size_range

        # r
        aspect_ratio = 0.3 + random_numbers[1] * (1.0 / 0.3 - 0.3)

        # a, b
        block_height = torch.sqrt(block_size * aspect_ratio)
        block_width = torch.sqrt(block_size / aspect_ratio)

        # t, l
        block_upper_float = random_numbers[2]
        block_upper_float = block_upper_float * (input_height - block_height)

        block_left_float = random_numbers[3]
        block_left_float = block_left_float * (input_width - block_width)

        # i, j
        block_upper = int(block_upper_float)
        block_lower = math.ceil(block_upper_float + block_height)

        block_left = int(block_left_float)
        block_right = math.ceil(block_left_float + block_width)

        return block_upper, block_lower, block_left, block_right

    def generate_mask(self, batch_size, seq_length):
        target_mask_size = int(self.masking_ratio * seq_length)

        # Assuming square input images
        num_patches_height = int(math.sqrt(seq_length))
        num_patches_width = seq_length // num_patches_height

        # Constant
        min_block_size = 16

        # Collect the results for the whole minibatch into this tensor
        full_mask = torch.zeros((batch_size, seq_length))

        for batch_num in range(batch_size):
            mask = torch.zeros((num_patches_height, num_patches_width), dtype=torch.bool)

            remaining_tokens_to_mask = target_mask_size

            while remaining_tokens_to_mask > 0:
                max_block_size = max(min_block_size, remaining_tokens_to_mask)

                block_upper, block_lower, block_left, block_right = \
                    self.generate_mask_block(
                        num_patches_height,
                        num_patches_width,
                        min_block_size,
                        max_block_size)

                # mask patches: [i_low, i_high), [j_low, j_high)
                mask[block_upper:block_lower, block_left:block_right] = True

                current_mask_size = int(mask.sum())
                remaining_tokens_to_mask = target_mask_size - current_mask_size

            mask = mask.reshape((seq_length,))

            full_mask[batch_num] = mask

        return full_mask

    def forward(self, x: Tensor, blocks) -> Tuple[Tensor, Tensor]:
        """
        :param x: tensor to reorder (B, L, D)
        :return: x with elements reordered along the second dimension (B, L, D)
        :return: the indexes used to reorder x (B, L, D)
        """

        batch_size, seq_length, num_feature_dimensions = x.shape

        if blocks is None:
            mask = self.generate_mask(batch_size, seq_length)
        else:
            block_indices = torch.randint(low=0, high=blocks.shape[0], size=tuple([batch_size]))
            mask = blocks[block_indices]

        sorted_indices = torch.argsort(mask, dim=1).to(x.device)
        expanded_sorted_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, num_feature_dimensions)
        reordered_x = torch.gather(x, dim=1, index=expanded_sorted_indices)

        return reordered_x, expanded_sorted_indices


class ShufflePatches(nn.Module):
    """
    Shuffle Class:
    Shuffles the input along the second dimension independently for each item in the batch, returning the shuffled
    input tensor along with the indices describing the shuffle.
    """

    def __init__(self):
        super(ShufflePatches, self).__init__()

    def forward(self, x: Tensor, *args) -> Tuple[Tensor, Tensor]:
        """
        :param x: tensor to shuffle (B, L, D)
        :return: x with elements independently shuffled along the second dimension (B, L, D)
        :return: the indexes used to shuffle x (B, L, D)
        """
        indices = torch.argsort(torch.rand(x.shape[:2], device=x.device), dim=1) \
            .unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return torch.gather(x, dim=1, index=indices), indices


class SwapAxes(nn.Module):
    """
    SwapAxes Class:
    Swap axis as a layer-compatible operation.
    """

    def __init__(self, dim0: int, dim1: int):
        super(SwapAxes, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.swapaxes(x, self.dim0, self.dim1)


class Unfold(nn.Module):
    def __init__(self, patch_size: int, d_model: int = 0, which_impl: str = 'unfold'):
        super(Unfold, self).__init__()
        if which_impl == 'unfold':
            self.unfolder = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        elif which_impl == 'conv':
            if d_model is None:
                d_model = 3 * patch_size ** 2
            self.unfolder = nn.Sequential(
                nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size, bias=False),
                nn.Flatten(2)
            )
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unfolder(x)

    @property
    def kernel_size(self) -> int:
        return self.patch_size

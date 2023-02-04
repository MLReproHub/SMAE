import cProfile

import torch

from model.layer import ReorderToBlockWiseMask


def profile(f, filename):
    pr = cProfile.Profile()
    pr.enable()
    f()
    pr.disable()
    pr.dump_stats(filename)


def prepare_blocks(model, sample, device, n=1000):
    reorder = ReorderToBlockWiseMask(model.masking_ratio)
    with torch.no_grad():
        patches, _ = model.tokenize(sample)
    bs, seq_len, embed_dim = patches.shape
    blocks = reorder.generate_mask(n, seq_len)
    return blocks.to(device)

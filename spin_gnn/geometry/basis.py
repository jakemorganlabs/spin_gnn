# gaussian radial basis expansion of invariant scalars. reference: Schütt et al.
# 2017 (SchNet) expand distances on a grid of gaussians so a network can read a
# sharp threshold as a linear combination of bumps. every input here is already
# an invariant scalar, so the expansion is invariant column by column.
# flow:
# 1. centers sit on an even grid over [lo, hi], k of them.
# 2. the width is the grid spacing, so neighboring bumps overlap at exp(-1/2).
# 3. the output stacks one bump per center on a new trailing axis.

import torch
from torch import Tensor


def gaussian_rbf(x: Tensor, lo: float, hi: float, k: int) -> Tensor:
    # step 1: the grid must be non-degenerate and carry at least two centers.
    assert k >= 2, f"k must be at least 2, got {k}"
    assert hi > lo, f"hi must exceed lo, got lo={lo} hi={hi}"
    centers = torch.linspace(lo, hi, k, dtype=x.dtype, device=x.device)
    width = (hi - lo) / (k - 1)
    gamma = 1.0 / (2.0 * width * width)

    # step 2: one bump per center; values lie in (0, 1] and peak at the centers.
    out = torch.exp(-gamma * (x.unsqueeze(-1) - centers) ** 2)
    assert out.shape == (*x.shape, k)
    assert bool(((out >= 0.0) & (out <= 1.0)).all())
    return out

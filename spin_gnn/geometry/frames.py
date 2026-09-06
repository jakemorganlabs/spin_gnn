# rotation frames: Haar sampling on SO(3), the octahedral group, and application.
# flow:
# 1. sample Haar rotations by QR of a gaussian matrix with the sign fix.
# 2. enumerate the 24 rotations of the cube as signed permutations with det +1.
# 3. apply rotations to plain 3-vectors and to (3, c) channel tensors.
# 4. every function asserts its requires on entry and its ensures before return.

import torch
from torch import Tensor

_OCTAHEDRAL_CACHE: dict[torch.dtype, Tensor] = {}


def _few_ulps(dtype: torch.dtype) -> float:
    # contract slop: three-axis accumulation plus headroom, never a magic literal.
    return float(torch.finfo(dtype).eps * (3 * 3 * 3 * 2 * 2))


def random_rotation(
    n: int, generator: torch.Generator, dtype: torch.dtype = torch.float64
) -> Tensor:
    assert n >= 1, "n must be at least 1"
    # step 1: draw a gaussian matrix and factor it.
    gauss = torch.randn(n, 3, 3, generator=generator, dtype=dtype)
    q, r_upper = torch.linalg.qr(gauss)

    # step 2: fix the column signs so q is Haar instead of QR-biased.
    diag = torch.diagonal(r_upper, dim1=-2, dim2=-1)
    q = q * diag.sign().unsqueeze(-1)

    # step 3: flip the last column where the determinant landed on -1.
    det = torch.linalg.det(q)
    flip = det < 0
    last = q[..., 2]
    q[..., 2] = torch.where(flip.unsqueeze(-1), -last, last)

    # step 4: a proper rotation has determinant +1 and orthonormal columns.
    tol = _few_ulps(dtype)
    eye = torch.eye(3, dtype=dtype).expand(n, 3, 3)
    dets = torch.linalg.det(q)
    assert bool(((dets - 1).abs() <= tol).all()), "det(R) must be 1"
    assert bool(((q.transpose(-1, -2) @ q - eye).abs() <= tol).all()), "R^T R must be the identity"
    return q


def _build_octahedral(dtype: torch.dtype) -> Tensor:
    # step 1: walk all signed permutation matrices and keep determinant +1.
    from itertools import permutations, product

    mats = []
    for perm in permutations((0, 1, 2)):
        for signs in product((1, -1), repeat=3):
            m = torch.zeros(3, 3, dtype=torch.float64)
            for axis in range(3):
                m[axis, perm[axis]] = float(signs[axis])
            if float(torch.linalg.det(m)) > 0:
                mats.append(m)

    stacked = torch.stack(mats).to(dtype)
    assert stacked.shape == (24, 3, 3), "the octahedral group has exactly 24 rotations"

    # step 2: every entry is a sign or a zero, and the matrices are distinct.
    assert bool(((stacked.abs() == 0) | (stacked.abs() == 1)).all())
    identity = torch.eye(3, dtype=dtype)
    gaps = (stacked.unsqueeze(0) - stacked.unsqueeze(1)).abs().amax(dim=(-1, -2))
    same = gaps <= _few_ulps(dtype)
    matches = same & ~torch.eye(24, dtype=torch.bool)
    assert not bool(matches.any()), "the 24 rotations must be pairwise distinct"

    # step 3: closedness under products, and the identity matrix present.
    products = stacked.unsqueeze(1) @ stacked.unsqueeze(0)
    gaps = (products.unsqueeze(2) - stacked.reshape(1, 1, 24, 3, 3)).abs().amax(dim=(-1, -2))
    assert bool((gaps.amin(dim=-1) <= _few_ulps(dtype)).all()), "the group must be closed"
    identity = torch.eye(3, dtype=dtype)
    assert bool((((stacked - identity).abs().amax(dim=(-1, -2))) <= _few_ulps(dtype)).any()), (
        "the identity rotation must be present"
    )
    return stacked


def octahedral_rotations(dtype: torch.dtype = torch.float64) -> Tensor:
    # built once per dtype, cached at module level.
    if dtype not in _OCTAHEDRAL_CACHE:
        _OCTAHEDRAL_CACHE[dtype] = _build_octahedral(dtype)
    out = _OCTAHEDRAL_CACHE[dtype]
    assert out.shape == (24, 3, 3)
    tol = _few_ulps(dtype)
    dets = torch.linalg.det(out)
    assert bool(((dets - 1).abs() <= tol).all())
    return out


def rotate_vectors(rotation: Tensor, v: Tensor) -> Tensor:
    assert rotation.shape[-2:] == (3, 3), (
        f"rotation must end in (3, 3), got {tuple(rotation.shape)}"
    )
    assert v.shape[-1] == 3, f"v must have last dimension 3, got {tuple(v.shape)}"
    assert rotation.shape[0] == v.shape[0], "rotation and v must share the batch dim"
    # step 1: apply R to the last axis, broadcasting over extra middle axes.
    if v.dim() == 2:
        out = torch.einsum("bij,bj->bi", rotation, v)
    elif v.dim() == 3:
        out = torch.einsum("bij,bnj->bni", rotation, v)
    else:
        dims = "abcdefghijklmnopqrstuvwxyz"
        mid = dims[: v.dim() - 2]
        out = torch.einsum(f"bij,b{mid}j->b{mid}i", rotation, v)
    # step 2: a rotation preserves the norm of every vector.
    assert out.shape == v.shape
    tol = _few_ulps(v.dtype)
    before = torch.sqrt((v * v).sum(-1))
    after = torch.sqrt((out * out).sum(-1))
    gap = (after - before).abs()
    slack = tol * (1 + before).clamp(max=2).clamp(min=1)
    assert bool((gap <= slack).all()), "norms must survive a rotation"
    return out


def rotate_channels(rotation: Tensor, v: Tensor) -> Tensor:
    assert rotation.shape[-2:] == (3, 3), (
        f"rotation must end in (3, 3), got {tuple(rotation.shape)}"
    )
    assert v.ndim == 4, f"v must have shape (B, N, 3, c), got {tuple(v.shape)}"
    assert v.shape[-2] == 3, f"the rotated axis must be the 3-axis, got {tuple(v.shape)}"
    # step 1: contract the rotation against axis -2, leaving channels alone.
    out = torch.einsum("bij,bnjc->bnic", rotation, v)
    # step 2: norms along the rotated axis are unchanged.
    assert out.shape == v.shape
    tol = _few_ulps(v.dtype)
    before = torch.sqrt((v * v).sum(-2))
    after = torch.sqrt((out * out).sum(-2))
    gap = (after - before).abs()
    slack = tol * (1 + before).clamp(max=2).clamp(min=1)
    assert bool((gap <= slack).all())
    return out

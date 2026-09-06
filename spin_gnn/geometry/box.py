# box constraint and pairwise geometry for points in the axis-aligned box.
# flow:
# 1. define assertion helpers shared by every geometry module.
# 2. define safe norms and normalizers that never divide by a raw norm.
# 3. define box projection, reflection, and wall or coordinate features.
# 4. define pairwise displacement, distance, and direction over batched point sets.
# 5. every function asserts its requires on entry and its ensures before return.

import torch
from torch import Tensor

from spin_gnn.constants import BOX, EPS


def assert_vec3(t: Tensor, name: str) -> None:
    # entry check for anything that must be a 3-vector on its last axis.
    assert t.shape[-1] == 3, f"{name} must have last dimension 3, got shape {tuple(t.shape)}"


def assert_finite(t: Tensor, name: str) -> None:
    # fail loud on nan or inf before it poisons downstream geometry.
    assert bool(torch.isfinite(t).all()), f"{name} must be finite"


def _extent_tensor(box: tuple[float, float, float], ref: Tensor) -> Tensor:
    # box extent as a broadcastable 3-vector on the same device and dtype.
    for extent in box:
        assert extent > 0, f"box extents must be positive, got {box}"
    return torch.tensor(box, dtype=ref.dtype, device=ref.device)


def _assert_in_box(x: Tensor, box: tuple[float, float, float]) -> None:
    extent = _extent_tensor(box, x)
    tol = 1e-6 if x.dtype == torch.float64 else 1e-4
    assert bool(((x >= -tol) & (x <= extent + tol)).all()), f"x must lie inside the box {box}"


def safe_norm(v: Tensor, eps: float = EPS) -> Tensor:
    # step 1: require a 3-vector and a positive stabilizer.
    assert_vec3(v, "v")
    assert eps > 0, "eps must be positive"
    n = torch.sqrt((v * v).sum(dim=-1) + eps)
    # step 2: the stabilizer keeps the result at or above sqrt(eps).
    assert n.shape == v.shape[:-1]
    assert bool((n >= torch.sqrt(torch.tensor(eps, dtype=n.dtype, device=n.device))).all())
    return n


def normalize(v: Tensor, eps: float = EPS) -> Tensor:
    # step 1: require a 3-vector.
    assert_vec3(v, "v")
    # step 2: divide by the stabilized norm, never the raw norm.
    out = v / safe_norm(v, eps).unsqueeze(-1)
    assert out.shape == v.shape
    return out


def project_to_box(x: Tensor, box: tuple[float, float, float] = BOX) -> Tensor:
    # step 1: clamp each coordinate to its axis extent.
    assert_vec3(x, "x")
    extent = _extent_tensor(box, x)
    out = torch.clamp(x, torch.zeros_like(extent), extent)
    # step 2: idempotent operation, points inside return unchanged.
    _assert_in_box(out, box)
    return out


def reflect_into_box(
    x: Tensor, delta: Tensor, box: tuple[float, float, float] = BOX
) -> tuple[Tensor, Tensor]:
    assert_vec3(x, "x")
    assert_vec3(delta, "delta")
    assert x.shape == delta.shape, (
        f"x and delta must share shape, got {tuple(x.shape)} and {tuple(delta.shape)}"
    )
    extent = _extent_tensor(box, x)
    _assert_in_box(x, box)
    # step 1: at most one reflection each direction per axis, so cap displacement.
    assert bool((delta.abs() < (extent + extent)).all()), (
        "abs(delta) must stay below twice the box extent on every axis"
    )

    raw = x + delta

    # step 2: reflect across the far wall where the raw move overshoots.
    over_far = raw > extent
    over_far_shift = raw - extent
    x_far = extent.expand_as(x).clone()
    x_far[over_far] = (extent - over_far_shift)[over_far]

    # step 3: reflect across the near wall where the raw move undershoots.
    # a move cannot trip both walls because abs(delta) is below two extents.
    near_val = torch.clamp(-raw, min=0.0)
    x_step = torch.where(over_far, x_far, torch.clamp(raw, min=0.0))
    step_wall = near_val > 0

    # step 4: any wall return negates the displacement component on that axis.
    wall = over_far | step_wall
    new_delta = torch.where(wall, -delta, delta)
    x_out = torch.where(wall, x_step, raw)

    # step 5: numerical residue at a wall return sits exactly on the boundary.
    x_out = torch.clamp(x_out, torch.zeros_like(extent), extent)

    _assert_in_box(x_out, box)
    tol = 1e-9 if x.dtype == torch.float64 else 1e-5
    assert bool(((new_delta.abs() - delta.abs()).abs() <= tol).all()), (
        "abs(delta) must be preserved per axis"
    )
    inside = ((raw >= 0) & (raw <= extent)).all(dim=-1)
    if bool(inside.all()):
        assert bool(((x_out - raw).abs() <= tol).all())
        assert bool(((new_delta - delta).abs() <= tol).all())
    return x_out, new_delta


def wall_features(x: Tensor, box: tuple[float, float, float] = BOX) -> Tensor:
    assert_vec3(x, "x")
    extent = _extent_tensor(box, x)
    _assert_in_box(x, box)
    # step 1: distance to the near and far wall per axis, interleaved as
    # (x, W-x, y, H-y, z, D-z).
    near = x
    far = extent - x
    feats = torch.stack((near, far), dim=-1).flatten(start_dim=-2)
    # step 2: distances are non-negative and opposite entries sum to the extent.
    assert feats.shape[-1] == 6
    assert bool((feats >= 0).all())
    pairs = feats.reshape(*feats.shape[:-1], 3, 2)
    sums = pairs.sum(dim=-1)
    assert bool(((sums - extent) == 0).all()), "opposite wall features must sum to the extent"
    return feats


def box_coords(x: Tensor, box: tuple[float, float, float] = BOX) -> Tensor:
    assert_vec3(x, "x")
    extent = _extent_tensor(box, x)
    _assert_in_box(x, box)
    # step 1: per-axis position as a fraction of the extent.
    frac = x / extent
    # step 2: quadratic term is largest mid-box, zero at the walls.
    feat = frac * (1 - frac)
    out = torch.cat((frac, feat), dim=-1)
    assert out.shape[-1] == 6
    assert bool(((out[..., :3] >= 0) & (out[..., :3] <= 1)).all())
    assert bool(((out[..., 3:] >= 0) & (out[..., 3:] <= 0.25)).all())
    return out


def pairwise_displacement(x: Tensor) -> Tensor:
    # step 1: broadcast the batched point set into all ordered pairs.
    assert x.ndim == 3, f"x must have shape (B, N, 3), got {tuple(x.shape)}"
    assert_vec3(x, "x")
    r = x.unsqueeze(2) - x.unsqueeze(1)
    # step 2: antisymmetric with an exactly zero diagonal.
    assert r.shape == (x.shape[0], x.shape[1], x.shape[1], 3)
    assert bool((r + r.transpose(1, 2) == 0).all())
    diag = r.diagonal(dim1=1, dim2=2)
    assert bool((diag == 0).all())
    return r


def pairwise_distance(r: Tensor, eps: float = EPS) -> Tensor:
    # step 1: norm each displacement with the stabilized norm.
    assert_vec3(r, "r")
    d = safe_norm(r, eps)
    # step 2: distance is symmetric in (i, j) by construction.
    assert d.shape == r.shape[:-1]
    return d


def pairwise_direction(r: Tensor, eps: float = EPS) -> Tensor:
    # step 1: normalize each displacement with the stabilized norm.
    assert_vec3(r, "r")
    d_hat = normalize(r, eps)
    # step 2: direction is antisymmetric in (i, j) by construction.
    assert d_hat.shape == r.shape
    return d_hat

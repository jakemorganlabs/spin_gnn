# autograd checks on every differentiable geometry function.
# flow:
# 1. sample inputs away from documented seams.
# 2. run torch.autograd.gradcheck in float64 on each function.

import torch
from torch.autograd import gradcheck

from spin_gnn.constants import OMEGA_MAX
from spin_gnn.geometry.box import (
    box_coords,
    normalize,
    pairwise_direction,
    pairwise_distance,
    safe_norm,
    wall_features,
)
from spin_gnn.geometry.spin import (
    angle_features,
    angular_momentum,
    axis_alignment,
    phase_difference,
    softplus_inv,
    spine_to_line,
    wrap_angle,
)

F64 = torch.float64


def _vec(seed: int, scale: float = 1.0) -> torch.Tensor:
    # a well-conditioned float64 3-vector with norm well above zero.
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(3, generator=g, dtype=F64)
    v = v / torch.linalg.norm(v) * (0.5 + scale)
    return v.requires_grad_(True)


def _point(seed: int) -> torch.Tensor:
    # a point at least 0.05 from every wall.
    g = torch.Generator().manual_seed(seed)
    return (0.1 + 0.8 * torch.rand(3, generator=g, dtype=F64)).requires_grad_(True)


def _angle(seed: int) -> torch.Tensor:
    # an angle at least 0.1 from the pi seam.
    g = torch.Generator().manual_seed(seed)
    return (0.31 + 1.7 * torch.rand(1, generator=g, dtype=F64)).squeeze(0).requires_grad_(True)


def _pair(seed_i: int, seed_j: int) -> tuple[torch.Tensor, torch.Tensor]:
    # two points far enough apart that direction is smooth.
    a = _point(seed_i)
    b = _point(seed_j)
    if torch.linalg.norm(a.detach() - b.detach()) < 0.3:
        b = b.detach().add(0.4).clamp(0.1, 0.9).requires_grad_(True)
    return a, b


def test_gradcheck_safe_norm() -> None:
    assert gradcheck(safe_norm, (_vec(0),), raise_exception=True)


def test_gradcheck_normalize() -> None:
    assert gradcheck(normalize, (_vec(1),), raise_exception=True)


def test_gradcheck_pairwise_distance() -> None:
    a, b = _pair(2, 3)
    assert gradcheck(lambda x: pairwise_distance(x - b), (a,), raise_exception=True)


def test_gradcheck_pairwise_direction() -> None:
    a, b = _pair(4, 5)
    assert gradcheck(lambda x: pairwise_direction(x - b), (a,), raise_exception=True)


def test_gradcheck_angle_features() -> None:
    assert gradcheck(angle_features, (_angle(6),), raise_exception=True)


def test_gradcheck_phase_difference() -> None:
    assert gradcheck(phase_difference, (_angle(7), _angle(8)), raise_exception=True)


def test_gradcheck_wrap_angle() -> None:
    # wrapped away from the seam so the gradient is a clean 1.
    assert gradcheck(wrap_angle, (_angle(9),), raise_exception=True)


def test_gradcheck_axis_alignment() -> None:
    assert gradcheck(axis_alignment, (_vec(10), _vec(11)), raise_exception=True)


def test_gradcheck_spine_to_line() -> None:
    assert gradcheck(spine_to_line, (_vec(12), _vec(13)), raise_exception=True)


def test_gradcheck_angular_momentum() -> None:
    s = torch.tensor(0.7, dtype=F64, requires_grad=True)
    omega = torch.tensor(0.4 * OMEGA_MAX, dtype=F64, requires_grad=True)
    u = _vec(14)
    assert gradcheck(angular_momentum, (s, omega, u), raise_exception=True)


def test_gradcheck_softplus_inv() -> None:
    s = torch.tensor(0.9, dtype=F64, requires_grad=True)
    assert gradcheck(softplus_inv, (s,), raise_exception=True)


def test_gradcheck_box_coords() -> None:
    assert gradcheck(box_coords, (_point(15),), raise_exception=True)


def test_gradcheck_wall_features() -> None:
    assert gradcheck(wall_features, (_point(16),), raise_exception=True)

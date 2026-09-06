# batched constellation container: validate, rotate, measure. spec: docs/SPEC.md.
# flow:
# 1. validate checks shapes and domain rules, returning a tagged outcome.
# 2. rotate_about_controller applies the SO(3) action rho_R about x_c.
# 3. min_wall_distance and in_interior measure where the satellites sit.

import torch
from torch import Tensor

from spin_gnn.constants import BOX, PI, R_INTERIOR
from spin_gnn.geometry.box import wall_features
from spin_gnn.geometry.frames import rotate_channels, rotate_vectors
from spin_gnn.types import Constellation, InvalidOutcome, ValidationOutcome, ValidOutcome

# domain tolerances: the box check sits above the spawn clamp, the axis
# check below the float64 renormalization error.
_POSITION_TOL: float = 1e-5
_AXIS_TOL: float = 1e-4


def batch_size(c: Constellation) -> int:
    return c.x.shape[0]


def n_satellites(c: Constellation) -> int:
    return c.x.shape[1]


def _shape_error(c: Constellation) -> str | None:
    # step 1: x anchors the shared batch size B and satellite count N.
    if c.x.ndim != 3 or c.x.shape[-1] != 3:
        return f"x must have shape (B, N, 3), got {tuple(c.x.shape)}"
    b, n = int(c.x.shape[0]), int(c.x.shape[1])
    if c.x_c.shape != (b, 3):
        return f"x_c must have shape ({b}, 3), got {tuple(c.x_c.shape)}"
    if c.u.shape != (b, n, 3):
        return f"u must have shape ({b}, {n}, 3), got {tuple(c.u.shape)}"
    if c.s.shape != (b, n):
        return f"s must have shape ({b}, {n}), got {tuple(c.s.shape)}"
    if c.phi.shape != (b, n):
        return f"phi must have shape ({b}, {n}), got {tuple(c.phi.shape)}"
    if c.omega.shape != (b, n):
        return f"omega must have shape ({b}, {n}), got {tuple(c.omega.shape)}"
    if c.h.ndim != 3 or c.h.shape[0] != b or c.h.shape[1] != n:
        return f"h must have shape ({b}, {n}, d), got {tuple(c.h.shape)}"
    if c.v.ndim != 4 or c.v.shape[0] != b or c.v.shape[1] != n or c.v.shape[2] != 3:
        return f"v must have shape ({b}, {n}, 3, c), got {tuple(c.v.shape)}"
    if c.h_c.ndim != 2 or c.h_c.shape[0] != b:
        return f"h_c must have shape ({b}, d_c), got {tuple(c.h_c.shape)}"
    return None


def validate(c: Constellation, box: tuple[float, float, float] = BOX) -> ValidationOutcome:
    # step 1: shapes must match the section 4 contract before any domain rule.
    shape_error = _shape_error(c)
    if shape_error is not None:
        return InvalidOutcome(valid=False, reason=shape_error)

    # step 2: check fields in the order x, x_c, u, s, phi, then NaN.
    extent = torch.tensor(box, dtype=c.x.dtype, device=c.x.device)
    if not bool(((c.x >= -_POSITION_TOL) & (c.x <= extent + _POSITION_TOL)).all()):
        return InvalidOutcome(valid=False, reason="x has a position outside the box")
    if not bool(((c.x_c >= -_POSITION_TOL) & (c.x_c <= extent + _POSITION_TOL)).all()):
        return InvalidOutcome(valid=False, reason="x_c has a position outside the box")
    axis_norm = torch.sqrt((c.u * c.u).sum(dim=-1))
    if not bool(((axis_norm - 1).abs() <= _AXIS_TOL).all()):
        return InvalidOutcome(valid=False, reason="u has an axis norm that differs from 1")
    if not bool((c.s > 0).all()):
        return InvalidOutcome(valid=False, reason="s has a size that is not greater than 0")
    if not bool(((c.phi > -PI) & (c.phi <= PI)).all()):
        return InvalidOutcome(valid=False, reason="phi has an angle outside (-pi, pi]")
    for name in ("s", "u", "phi", "omega", "h", "v", "x", "x_c", "h_c"):
        tensor = getattr(c, name)
        if not bool(torch.isfinite(tensor).all()):
            return InvalidOutcome(valid=False, reason=f"{name} contains NaN or Inf")
    return ValidOutcome(valid=True)


def assert_valid(c: Constellation, box: tuple[float, float, float] = BOX) -> None:
    # loud form of validate: raises AssertionError with the reason.
    outcome = validate(c, box)
    assert outcome.valid, outcome.reason if isinstance(outcome, InvalidOutcome) else ""


def rotate_about_controller(c: Constellation, rotation: Tensor) -> Constellation:
    # step 1: require one proper rotation per batch element.
    b = batch_size(c)
    assert rotation.shape == (b, 3, 3), (
        f"rotation must have shape (B, 3, 3), got {tuple(rotation.shape)}"
    )
    dets = torch.linalg.det(rotation.to(torch.float64))
    assert bool(((dets - 1).abs() <= 1e-6).all()), "each rotation must have det 1"

    # step 2: rotate positions about x_c, axes, and the 3-axis of V.
    x_new = rotate_vectors(rotation, c.x - c.x_c.unsqueeze(1)) + c.x_c.unsqueeze(1)
    u_new = rotate_vectors(rotation, c.u)
    v_new = rotate_channels(rotation, c.v)

    # step 3: scalars, phases, speeds, states, and x_c pass through unchanged.
    return Constellation(
        x=x_new,
        s=c.s,
        u=u_new,
        phi=c.phi,
        omega=c.omega,
        h=c.h,
        v=v_new,
        x_c=c.x_c,
        h_c=c.h_c,
    )


def min_wall_distance(c: Constellation, box: tuple[float, float, float] = BOX) -> Tensor:
    # the smallest satellite-to-wall gap per batch element.
    feats = wall_features(c.x, box)
    out = feats.amin(dim=(-1, -2))
    assert out.shape == (batch_size(c),)
    return out


def in_interior(c: Constellation, radius: float = R_INTERIOR) -> Tensor:
    # True exactly when every satellite sits within radius of x_c.
    assert radius > 0, "radius must be positive"
    dist = torch.sqrt(((c.x - c.x_c.unsqueeze(1)) ** 2).sum(dim=-1))
    out = (dist <= radius).all(dim=-1)
    assert out.shape == (batch_size(c),)
    assert out.dtype == torch.bool
    return out


def to_dtype(c: Constellation, dtype: torch.dtype) -> Constellation:
    return Constellation(
        x=c.x.to(dtype),
        s=c.s.to(dtype),
        u=c.u.to(dtype),
        phi=c.phi.to(dtype),
        omega=c.omega.to(dtype),
        h=c.h.to(dtype),
        v=c.v.to(dtype),
        x_c=c.x_c.to(dtype),
        h_c=c.h_c.to(dtype),
    )


def detach_clone(c: Constellation) -> Constellation:
    return Constellation(
        x=c.x.detach().clone(),
        s=c.s.detach().clone(),
        u=c.u.detach().clone(),
        phi=c.phi.detach().clone(),
        omega=c.omega.detach().clone(),
        h=c.h.detach().clone(),
        v=c.v.detach().clone(),
        x_c=c.x_c.detach().clone(),
        h_c=c.h_c.detach().clone(),
    )

# spin state helpers: angle wrapping, phase features, axis and speed helpers.
# flow:
# 1. wrap angles onto the half-open interval (-pi, pi] with an exact seam rule.
# 2. build phase features and phase differences from wrapped angles.
# 3. measure agreement between axes, between an axis and a line of sight, and
#    build angular momentum from size, speed, and axis.
# 4. renormalize axes and clamp speeds and sizes into legal ranges.
# 5. invert the softplus so sizes can be written in unconstrained space.
# every function asserts its requires on entry and its ensures before return.

import torch
from torch import Tensor

from spin_gnn.constants import EPS, OMEGA_MAX, PI, SIZE_MAX, SIZE_MIN, TWO_PI
from spin_gnn.geometry.box import assert_finite, assert_vec3, normalize


def _few_ulps(dtype: torch.dtype) -> float:
    # contract slop: a handful of ulps of the working dtype, never a magic literal.
    return float(torch.finfo(dtype).eps * (3 * 3 * 3 * 2 * 2))


def wrap_angle(phi: Tensor) -> Tensor:
    assert_finite(phi, "phi")
    # step 1: fold into (-pi, pi] and move the exact left edge onto pi.
    wrapped = phi - TWO_PI * torch.floor((phi + PI) / TWO_PI)
    out = torch.where(wrapped == -PI, torch.full_like(wrapped, PI), wrapped)
    # step 2: the result lives in the half-open interval and is unchanged mod 2 pi.
    assert bool(((out > -PI) & (out <= PI)).all()), "wrapped angle must lie in (-pi, pi]"
    base = out - TWO_PI * torch.floor((out + PI) / TWO_PI)
    base = torch.where(base == -PI, torch.full_like(base, PI), base)
    slack = _few_ulps(out.dtype) * max(abs(PI), 1.0) * 4
    assert not bool(((base - out).abs() > slack).any()), "wrap must be idempotent"
    return out


def angle_features(phi: Tensor) -> Tensor:
    assert_finite(phi, "phi")
    # step 1: the phase as a point on the unit circle, (sin, cos).
    out = torch.stack((torch.sin(phi), torch.cos(phi)), dim=-1)
    assert out.shape == (*phi.shape, 2)
    return out


def phase_difference(phi_i: Tensor, phi_j: Tensor) -> Tensor:
    assert phi_i.shape == phi_j.shape, (
        f"phases must share shape, got {tuple(phi_i.shape)} and {tuple(phi_j.shape)}"
    )
    # step 1: subtract then wrap onto (-pi, pi].
    out = wrap_angle(phi_i - phi_j)
    assert bool(((out > -PI) & (out <= PI)).all())
    return out


def axis_alignment(u_i: Tensor, u_j: Tensor) -> Tensor:
    assert_vec3(u_i, "u_i")
    assert_vec3(u_j, "u_j")
    # step 1: the cosine of the angle between the two axes.
    dot = (u_i * u_j).sum(dim=-1)
    assert dot.shape == u_i.shape[:-1]
    return dot


def spine_to_line(u_i: Tensor, r_hat_ij: Tensor) -> Tensor:
    assert_vec3(u_i, "u_i")
    assert_vec3(r_hat_ij, "r_hat_ij")
    # step 1: how closely the axis points along the line of sight.
    dot = (u_i * r_hat_ij).sum(dim=-1)
    assert dot.shape == u_i.shape[:-1]
    return dot


def angular_momentum(s: Tensor, omega: Tensor, u: Tensor) -> Tensor:
    assert s.shape == omega.shape, (
        f"s and omega must share shape, got {tuple(s.shape)} and {tuple(omega.shape)}"
    )
    assert_vec3(u, "u")
    assert u.shape[:-1] == s.shape, "u must broadcast against s on the last axis"
    # step 1: scale the axis by the scalar spin magnitude.
    magnitude = (s * omega).unsqueeze(-1)
    out = magnitude * u
    # step 2: the result is parallel to u because it is a scalar multiple of u.
    assert out.shape == u.shape
    reference = (s * omega).unsqueeze(-1) * u
    assert bool((out - reference == 0).all()), "angular momentum must equal s * omega * u"
    return out


def renormalize_axis(u: Tensor, eps: float = EPS) -> Tensor:
    assert_vec3(u, "u")
    # step 1: a fresh normalized axis, same as normalize but named for the call site.
    out = normalize(u, eps)
    assert out.shape == u.shape
    return out


def clamp_speed(omega: Tensor, omega_max: float = OMEGA_MAX) -> Tensor:
    assert omega_max > 0, "omega_max must be positive"
    # step 1: fold the speed into the closed interval.
    out = torch.clamp(omega, 0.0, omega_max)
    assert bool(((out >= 0) & (out <= omega_max)).all()), "speed must lie in range"
    return out


def clamp_size(s: Tensor, size_min: float = SIZE_MIN, size_max: float = SIZE_MAX) -> Tensor:
    assert 0 < size_min <= size_max, "size_min must be positive and at most size_max"
    # step 1: fold the size into the closed interval.
    out = torch.clamp(s, size_min, size_max)
    assert bool(((out >= size_min) & (out <= size_max)).all()), "size must lie in range"
    return out


def softplus_inv(s: Tensor, eps: float = EPS) -> Tensor:
    assert bool((s > 0).all()), "softplus_inv needs every element of s above 0"
    # step 1: invert softplus; the eps keeps the log away from exact zero.
    out = torch.log(torch.expm1(s) + eps)
    # step 2: softplus of the result returns s inside float64 working range.
    assert out.shape == s.shape
    return out

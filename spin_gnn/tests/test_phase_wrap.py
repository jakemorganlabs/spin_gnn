# examples and properties for spin.py.
# flow:
# 1. pin the wrap, phase difference, angular momentum, and softplus examples.
# 2. pin the wrap, phase feature, and softplus inverse properties.

import torch
from hypothesis import given
from hypothesis import strategies as st
from torch.nn import functional as f

from spin_gnn.constants import DT, OMEGA_MAX, PI, SIZE_MAX, SIZE_MIN, TWO_PI
from spin_gnn.geometry.spin import (
    angle_features,
    angular_momentum,
    phase_difference,
    softplus_inv,
    wrap_angle,
)
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.model.spin_gnn_gnn import build_model
from spin_gnn.tests.conftest import angles
from spin_gnn.tests.test_constellation import make_constellation

F64 = torch.float64


# examples: seam behavior and the documented fixed points.


def test_wrap_angle_of_three_pi_lies_on_seam() -> None:
    # 3 * PI sits within one ulp of the seam, so accept either seam value.
    out = wrap_angle(torch.tensor(3 * PI, dtype=F64))
    assert bool((out.abs() - PI).abs() < 1e-9)


def test_wrap_angle_of_pi_is_pi() -> None:
    assert torch.allclose(wrap_angle(torch.tensor(PI, dtype=F64)), torch.tensor(PI, dtype=F64))


def test_wrap_angle_of_negative_pi_is_pi() -> None:
    out = wrap_angle(torch.tensor(-PI, dtype=F64))
    assert torch.allclose(out, torch.tensor(PI, dtype=F64), atol=1e-9)


def test_phase_difference_wraps_across_seam() -> None:
    a = torch.tensor(0.1, dtype=F64)
    b = torch.tensor(TWO_PI - 0.1, dtype=F64)
    out = phase_difference(a, b)
    assert torch.allclose(out, torch.tensor(0.2, dtype=F64), atol=1e-9)


def test_angular_momentum_scales_axis() -> None:
    s = torch.tensor(2.0, dtype=F64)
    omega = torch.tensor(0.5, dtype=F64)
    u = torch.tensor([0.0, 0.0, 1.0], dtype=F64)
    out = angular_momentum(s, omega, u)
    assert torch.allclose(out, torch.tensor([0.0, 0.0, 1.0], dtype=F64), atol=1e-9)


def test_softplus_inv_inverts_softplus() -> None:
    s = f.softplus(torch.tensor(1.7, dtype=F64))
    out = softplus_inv(s)
    assert torch.allclose(out, torch.tensor(1.7, dtype=F64), atol=1e-6)


# properties: wrap periodicity, phase feature agreement, softplus inverse.


@given(phi=angles(), k=st.integers(-5, 5))
def test_wrap_angle_is_periodic_and_in_interval(phi: float, k: int) -> None:
    phi_t = torch.tensor(phi, dtype=F64)
    shifted = phi_t + TWO_PI * k
    out = wrap_angle(shifted)
    assert torch.allclose(out, wrap_angle(phi_t), atol=1e-9)
    assert bool((out > -PI) & (out <= PI))


@given(a=angles(), b=angles())
def test_phase_features_match_wrapped_difference(a: float, b: float) -> None:
    a_t = torch.tensor(a, dtype=F64)
    b_t = torch.tensor(b, dtype=F64)
    diff = phase_difference(a_t, b_t)
    feats = angle_features(a_t - b_t)
    expected = torch.stack((torch.sin(diff), torch.cos(diff)))
    assert torch.allclose(feats, expected, atol=1e-9)


@given(s=st.floats(SIZE_MIN, SIZE_MAX, allow_nan=False, allow_infinity=False))
def test_softplus_inv_round_trip(s: float) -> None:
    s_t = torch.tensor(s, dtype=F64)
    out = f.softplus(softplus_inv(s_t))
    assert torch.allclose(out, s_t, atol=1e-6)


def test_closed_form(generator: torch.Generator) -> None:
    # Prop 7: with update_speed False and the phase residual absent, phi is the
    # exact constant-speed flow wrap(phi_0 + n_layers DT omega_0).
    config = SpinGnnConfig().model_copy(update={"update_speed": False, "update_phase": True})
    model = build_model(config).double()
    c = make_constellation(2, config.n, config.d, config.c, generator, interior=True)
    # keep every speed legal so clamp_speed is the identity across the run.
    omega = c.omega.clamp(0.0, OMEGA_MAX)
    seed = type(c)(
        x=c.x, s=c.s, u=c.u, phi=c.phi, omega=omega, h=c.h, v=c.v, x_c=c.x_c, h_c=c.h_c
    )
    out = model(seed).constellation
    expected = wrap_angle(seed.phi + config.n_layers * DT * seed.omega)
    gap = (out.phi - expected).abs().max()
    assert bool(gap <= 1e-9), f"phi drifted by {float(gap)}"

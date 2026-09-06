# tests for the constellation container: examples and properties.
# flow:
# 1. make_constellation builds domain-true fixtures, interior or in-box.
# 2. examples cover validation failures, interior, and wall distance.
# 3. properties cover rotation round-trip, validate stability, size_pair.

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st

from spin_gnn.constants import (
    MARGIN,
    OMEGA_MAX,
    R_INTERIOR,
    SEED,
    SIZE_MAX,
    SIZE_MIN,
)
from spin_gnn.constellation import (
    assert_valid,
    in_interior,
    min_wall_distance,
    rotate_about_controller,
    validate,
)
from spin_gnn.geometry.frames import random_rotation
from spin_gnn.geometry.invariants import size_pair
from spin_gnn.types import Constellation, InvalidOutcome, ValidOutcome


def make_constellation(
    batch: int,
    n: int,
    d: int,
    c: int,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
    interior: bool = False,
) -> Constellation:
    # build a domain-true constellation; v is zero until a session assigns it.
    if interior:
        x_c = torch.full((batch, 3), 0.5, dtype=dtype)
        direction = torch.randn(batch, n, 3, generator=generator, dtype=dtype)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        # radii stay strictly inside the interior ball so float noise at the
        # boundary cannot flip in_interior or push a satellite out of the box.
        radius = 0.01 + (0.95 * R_INTERIOR - 0.01) * torch.rand(
            batch, n, 1, generator=generator, dtype=dtype
        )
        x = x_c.unsqueeze(1) + radius * direction
    else:
        x = MARGIN + (1 - 2 * MARGIN) * torch.rand(batch, n, 3, generator=generator, dtype=dtype)
        x_c = MARGIN + (1 - 2 * MARGIN) * torch.rand(batch, 3, generator=generator, dtype=dtype)
    u = torch.randn(batch, n, 3, generator=generator, dtype=dtype)
    u = u / u.norm(dim=-1, keepdim=True)
    phi = (torch.rand(batch, n, generator=generator, dtype=dtype) * 2 - 1) * torch.pi
    omega = (torch.rand(batch, n, generator=generator, dtype=dtype) * 2 - 1) * OMEGA_MAX
    log_size = torch.rand(batch, n, generator=generator, dtype=dtype)
    s = torch.exp(
        torch.log(torch.tensor(SIZE_MIN, dtype=dtype))
        + log_size * torch.log(torch.tensor(SIZE_MAX / SIZE_MIN, dtype=dtype))
    )
    h = torch.randn(batch, n, d, generator=generator, dtype=dtype)
    h_c = torch.randn(batch, 2 * d, generator=generator, dtype=dtype)
    v = torch.zeros(batch, n, 3, c, dtype=dtype)
    out = Constellation(x=x, s=s, u=u, phi=phi, omega=omega, h=h, v=v, x_c=x_c, h_c=h_c)
    assert_valid(out)
    return out


def _edited(c: Constellation, **fields: torch.Tensor) -> Constellation:
    # frozen dataclasses reject assignment; rebuild with the field swapped.
    data = dict(x=c.x, s=c.s, u=c.u, phi=c.phi, omega=c.omega, h=c.h, v=c.v, x_c=c.x_c, h_c=c.h_c)
    data.update(fields)
    return Constellation(**data)  # type: ignore[arg-type]


def test_validate_accepts_interior_fixture(generator: torch.Generator) -> None:
    c = make_constellation(2, 16, 4, 2, generator, interior=True)
    outcome = validate(c)
    assert isinstance(outcome, ValidOutcome)


def test_validate_rejects_position_outside_box(generator: torch.Generator) -> None:
    c = make_constellation(2, 16, 4, 2, generator, interior=True)
    bad = c.x.clone()
    bad[0, 3] = torch.tensor([1.2, 0.5, 0.5], dtype=c.x.dtype)
    outcome = validate(_edited(c, x=bad))
    assert isinstance(outcome, InvalidOutcome)
    assert "x" in outcome.reason


def test_validate_rejects_nonunit_axis(generator: torch.Generator) -> None:
    c = make_constellation(2, 16, 4, 2, generator, interior=True)
    bad = c.u.clone()
    bad[0, 0] = torch.tensor([2.0, 0.0, 0.0], dtype=c.u.dtype)
    outcome = validate(_edited(c, u=bad))
    assert isinstance(outcome, InvalidOutcome)
    assert "u" in outcome.reason


def test_validate_rejects_nonpositive_size(generator: torch.Generator) -> None:
    c = make_constellation(2, 16, 4, 2, generator, interior=True)
    bad = c.s.clone()
    bad[1, 5] = 0.0
    outcome = validate(_edited(c, s=bad))
    assert isinstance(outcome, InvalidOutcome)
    assert "s" in outcome.reason


def test_validate_rejects_phase_outside_half_turn(generator: torch.Generator) -> None:
    c = make_constellation(2, 16, 4, 2, generator, interior=True)
    bad = c.phi.clone()
    bad[0, 0] = 4.0
    outcome = validate(_edited(c, phi=bad))
    assert isinstance(outcome, InvalidOutcome)
    assert "phi" in outcome.reason


def test_validate_rejects_nan_in_state(generator: torch.Generator) -> None:
    c = make_constellation(2, 16, 4, 2, generator, interior=True)
    bad = c.h.clone()
    bad[0, 0, 0] = float("nan")
    outcome = validate(_edited(c, h=bad))
    assert isinstance(outcome, InvalidOutcome)
    assert "h" in outcome.reason


def test_assert_valid_raises_with_reason(generator: torch.Generator) -> None:
    c = make_constellation(2, 16, 4, 2, generator, interior=True)
    bad = c.s.clone()
    bad[0, 0] = -1.0
    try:
        assert_valid(_edited(c, s=bad))
    except AssertionError as err:
        assert "s" in str(err)
    else:
        raise AssertionError("assert_valid must raise on an invalid constellation")


def test_interior_fixture_is_interior(generator: torch.Generator) -> None:
    c = make_constellation(2, 16, 4, 2, generator, interior=True)
    assert bool(in_interior(c).all())


def test_satellite_past_interior_radius_marks_batch_outside(
    generator: torch.Generator,
) -> None:
    c = make_constellation(2, 16, 4, 2, generator, interior=True)
    moved = c.x.clone()
    moved[0, 0] = torch.tensor([0.99, 0.5, 0.5], dtype=c.x.dtype)
    flags = in_interior(_edited(c, x=moved))
    assert not bool(flags[0])
    assert bool(flags[1])


def test_min_wall_distance_matches_placed_satellite(generator: torch.Generator) -> None:
    c = make_constellation(1, 16, 4, 2, generator, interior=True)
    moved = c.x.clone()
    moved[0, 0] = torch.tensor([0.02, 0.5, 0.5], dtype=c.x.dtype)
    gap = min_wall_distance(_edited(c, x=moved))
    assert bool((gap[0] - 0.02).abs() <= 1e-9)


@given(st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=3))
def test_rotate_about_controller_round_trips_x_u_and_v(b_index: int, seed_offset: int) -> None:
    gen = torch.Generator().manual_seed(SEED + seed_offset * 131 + b_index)
    c = make_constellation(2, 8, 4, 2, gen, interior=True)
    rotation = random_rotation(2, gen)
    round_tripped = rotate_about_controller(
        rotate_about_controller(c, rotation), rotation.transpose(-1, -2)
    )
    assert bool(((round_tripped.x - c.x).abs() <= 1e-9).all())
    assert bool(((round_tripped.u - c.u).abs() <= 1e-9).all())
    assert bool(((round_tripped.v - c.v).abs() <= 1e-9).all())


@given(st.integers(min_value=1, max_value=3))
def test_validate_holds_before_and_after_rotation(seed_offset: int) -> None:
    gen = torch.Generator().manual_seed(SEED + 911 + seed_offset)
    c = make_constellation(2, 16, 4, 2, gen, interior=True)
    assert isinstance(validate(c), ValidOutcome)
    rotated = rotate_about_controller(c, random_rotation(2, gen))
    assert isinstance(validate(rotated), ValidOutcome)
    # the untouched fields are the very same tensors.
    assert rotated.s is c.s
    assert rotated.phi is c.phi
    assert rotated.omega is c.omega
    assert rotated.h is c.h
    assert rotated.h_c is c.h_c
    assert rotated.x_c is c.x_c


@given(
    st.floats(min_value=1e-6, max_value=100.0),
    st.floats(min_value=1e-6, max_value=100.0),
)
def test_size_pair_antisymmetry_and_sigma_sum(s_i: float, s_j: float) -> None:
    rho, sigma = size_pair(
        torch.tensor(s_i, dtype=torch.float64), torch.tensor(s_j, dtype=torch.float64)
    )
    rho_back, sigma_back = size_pair(
        torch.tensor(s_j, dtype=torch.float64), torch.tensor(s_i, dtype=torch.float64)
    )
    assert bool((rho + rho_back).abs() <= 1e-9)
    assert bool((sigma + sigma_back - 1).abs() <= 1e-9)


def test_to_dtype_and_detach_clone_preserve_values(generator: torch.Generator) -> None:
    c = make_constellation(2, 4, 4, 2, generator, interior=True)
    c32 = c  # keep the float64 fixture for comparison
    from spin_gnn.constellation import detach_clone, to_dtype

    f32 = to_dtype(c32, torch.float32)
    assert f32.x.dtype == torch.float32
    back = to_dtype(f32, torch.float64)
    assert np.allclose(back.x.numpy(), c.x.numpy(), atol=1e-6)
    cloned = detach_clone(c)
    assert cloned.x is not c.x
    assert bool((cloned.x == c.x).all())

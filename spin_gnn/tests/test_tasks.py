# tests for Tasks A, D, and E. spec: docs/SPEC.md.
# flow:
# 1. _edited swaps one field on the frozen container.
# 2. examples cover task A corruption, the task D pair vote, and the task E
#    free-flight hit count.
# 3. the property covers rotation invariance of tasks D and E with speeds
#    capped so the rollout stays inside the box.

import torch
from hypothesis import given
from hypothesis import strategies as st

from spin_gnn.constants import OMEGA_MAX, R_INTERIOR, SEED
from spin_gnn.constellation import (
    assert_valid,
    in_interior,
    rotate_about_controller,
    validate,
)
from spin_gnn.geometry.frames import random_rotation
from spin_gnn.geometry.spin import unit_axis_exact
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.train.tasks_synthetic import (
    corrupt_for_task_a,
    majority_spin_align,
    rollout_hit,
    sample_interior,
)
from spin_gnn.types import Constellation, ValidOutcome


def _edited(c: Constellation, **fields: torch.Tensor) -> Constellation:
    # frozen dataclasses reject assignment; rebuild with the field swapped.
    data = dict(
        x=c.x, s=c.s, u=c.u, phi=c.phi, omega=c.omega,
        h=c.h, v=c.v, x_c=c.x_c, h_c=c.h_c,
    )
    data.update(fields)
    return Constellation(**data)  # type: ignore[arg-type]


def test_task_a_noisy_valid_and_targets_sized(generator: torch.Generator) -> None:
    config = SpinGnnConfig()
    c = sample_interior(4, config, generator, dtype=torch.float64)
    noisy, targets = corrupt_for_task_a(c, generator)
    assert isinstance(validate(noisy), ValidOutcome)
    assert targets.shape == (4, 3)
    rel = c.x - c.x_c.unsqueeze(1)
    expected_d = rel.norm(dim=-1).mean(dim=-1)
    assert bool(((targets[:, 0] - expected_d).abs() <= 1e-9).all())


def test_task_a_targets_come_from_the_clean_cloud(generator: torch.Generator) -> None:
    # the corruption must not move the target: a noisy clone yields the same
    # three numbers as the clean constellation.
    config = SpinGnnConfig()
    c = sample_interior(2, config, generator, dtype=torch.float64)
    _, targets = corrupt_for_task_a(c, generator)
    rel = c.x - c.x_c.unsqueeze(1)
    line = unit_axis_exact(rel)
    expected_align = (c.u * line).sum(dim=-1).mean(dim=-1)
    expected_omega = c.omega.mean(dim=-1)
    assert bool(((targets[:, 1] - expected_align).abs() <= 1e-9).all())
    assert bool(((targets[:, 2] - expected_omega).abs() <= 1e-9).all())


def _two_satellite_scene(
    generator: torch.Generator,
    separation: float,
    u0: tuple[float, float, float],
    u1: tuple[float, float, float],
) -> torch.Tensor:
    # a two-satellite constellation, the pair `separation` apart. with n = 2
    # the only gated pair is (0, 1), so the vote is 1 exactly when its axes
    # align past TASK_D_ALIGN.
    config = SpinGnnConfig(n=2)
    c = sample_interior(1, config, generator, dtype=torch.float64)
    x = c.x.clone()
    x[0, 0] = c.x_c[0] + torch.tensor([0.15, 0.0, 0.0], dtype=c.x.dtype)
    x[0, 1] = x[0, 0] + torch.tensor([0.0, separation, 0.0], dtype=c.x.dtype)
    u = c.u.clone()
    u[0, 0] = torch.tensor(u0, dtype=c.u.dtype)
    u[0, 1] = torch.tensor(u1, dtype=c.u.dtype)
    return majority_spin_align(_edited(c, x=x, u=u))


def test_task_d_close_aligned_pair_votes_one(generator: torch.Generator) -> None:
    out = _two_satellite_scene(
        generator, separation=0.2, u0=(1.0, 0.0, 0.0), u1=(1.0, 0.0, 0.0)
    )
    assert out.tolist() == [1]


def test_task_d_close_orthogonal_pair_votes_zero(generator: torch.Generator) -> None:
    out = _two_satellite_scene(
        generator, separation=0.2, u0=(1.0, 0.0, 0.0), u1=(0.0, 1.0, 0.0)
    )
    assert out.tolist() == [0]


def test_task_e_one_speeding_satellite_scores_one(generator: torch.Generator) -> None:
    config = SpinGnnConfig()
    c = sample_interior(1, config, generator, dtype=torch.float64)
    x = c.x.clone()
    u = c.u.clone()
    omega = torch.zeros_like(c.omega)
    x[0, 0] = c.x_c[0] + torch.tensor([0.3, 0.0, 0.0], dtype=c.x.dtype)
    u[0, 0] = torch.tensor([-1.0, 0.0, 0.0], dtype=c.u.dtype)
    omega[0, 0] = OMEGA_MAX
    # park the rest far and static; far stays far because omega 0 never moves.
    x[0, 1:] = c.x_c[0] + torch.tensor([0.0, 0.0, R_INTERIOR * 0.9], dtype=c.x.dtype)
    u[0, 1:] = torch.tensor([0.0, 1.0, 0.0], dtype=c.u.dtype)
    out = rollout_hit(_edited(c, x=x, u=u, omega=omega))
    assert out.tolist() == [1]


def test_task_e_static_ring_scores_zero(generator: torch.Generator) -> None:
    config = SpinGnnConfig()
    c = sample_interior(1, config, generator, dtype=torch.float64)
    x = c.x.clone()
    direction = unit_axis_exact(
        torch.randn_like(x[0], generator=generator, dtype=c.x.dtype)
    )
    x[0] = c.x_c[0] + 0.34 * direction
    omega = torch.zeros_like(c.omega)
    out = rollout_hit(_edited(c, x=x, omega=omega))
    assert out.tolist() == [0]


@given(st.integers(min_value=1, max_value=6))
def test_tasks_d_and_e_rotation_invariant_on_interior(seed_offset: int) -> None:
    # Property 2. speeds are capped at 0.5 so travel stays under
    # DT * 0.5 * 4 = 0.5. a satellite at most 0.35 off the center then moves
    # no farther than 0.85, inside the wall at 1.0, so reflect_into_box never
    # fires and both labels are pure rotations of one another.
    generator = torch.Generator().manual_seed(SEED + 211 + seed_offset)
    config = SpinGnnConfig()
    c = sample_interior(4, config, generator, dtype=torch.float64)
    capped = _edited(c, omega=c.omega.clamp(-0.5, 0.5))
    assert_valid(capped)
    rotation = random_rotation(4, generator)
    rotated = rotate_about_controller(capped, rotation)
    assert bool(in_interior(rotated).all())
    assert bool((majority_spin_align(capped) == majority_spin_align(rotated)).all())
    assert bool((rollout_hit(capped) == rollout_hit(rotated)).all())

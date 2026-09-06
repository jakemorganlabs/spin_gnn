# tests for the SPEC predicate set: constructed cases and the count contract.
# flow:
# 1. examples pin near, corner, aligned, opp_spin, and the near count.
# 2. a phaselock example pins the joint aligned-plus-phase gate.
# 3. the property proves diagonals are False and counts match sums.

import torch
from hypothesis import given
from hypothesis import strategies as st

from spin_gnn.constants import SEED
from spin_gnn.model.readout import predicates
from spin_gnn.tests.test_constellation import make_constellation
from spin_gnn.types import Constellation


def _edited(c: Constellation, **fields: torch.Tensor) -> Constellation:
    # frozen dataclasses reject assignment; rebuild with the field swapped.
    data = dict(x=c.x, s=c.s, u=c.u, phi=c.phi, omega=c.omega, h=c.h, v=c.v, x_c=c.x_c, h_c=c.h_c)
    data.update(fields)
    return Constellation(**data)  # type: ignore[arg-type]


def test_predicate_dtypes_and_shapes(generator: torch.Generator) -> None:
    c = make_constellation(3, 16, 4, 2, generator, interior=True)
    p = predicates(c)
    for field in (p.near, p.fast, p.corner):
        assert field.shape == (3, 16)
        assert field.dtype == torch.bool
    for field in (p.aligned, p.phaselock, p.opp_spin):
        assert field.shape == (3, 16, 16)
        assert field.dtype == torch.bool
    for field in (p.n_near, p.n_fast_near, p.n_aligned_pairs):
        assert field.shape == (3,)
        assert field.dtype == torch.int64


def test_near_true_for_satellite_close_to_controller(generator: torch.Generator) -> None:
    c = make_constellation(1, 16, 4, 2, generator, interior=True)
    moved = c.x.clone()
    moved[0, 0] = c.x_c[0] + torch.tensor([0.1, 0.0, 0.0], dtype=c.x.dtype)
    p = predicates(_edited(c, x=moved))
    assert bool(p.near[0, 0])


def test_corner_true_for_satellite_at_wall(generator: torch.Generator) -> None:
    c = make_constellation(1, 16, 4, 2, generator, interior=True)
    moved = c.x.clone()
    moved[0, 0] = torch.tensor([0.01, 0.5, 0.5], dtype=c.x.dtype)
    p = predicates(_edited(c, x=moved))
    assert bool(p.corner[0, 0])


def test_aligned_true_for_shared_axis_and_diagonal_false(generator: torch.Generator) -> None:
    c = make_constellation(1, 16, 4, 2, generator, interior=True)
    axes = c.u.clone()
    axes[0, 1] = axes[0, 0]
    p = predicates(_edited(c, u=axes))
    assert bool(p.aligned[0, 0, 1])
    assert bool(p.aligned[0, 1, 0])
    assert not bool(p.aligned[0, 0, 0])


def test_opp_spin_true_for_opposite_speeds_on_shared_axis(
    generator: torch.Generator,
) -> None:
    c = make_constellation(1, 16, 4, 2, generator, interior=True)
    axes = c.u.clone()
    axes[0, 1] = axes[0, 0]
    speeds = c.omega.clone()
    speeds[0, 0] = 1.0
    speeds[0, 1] = -1.0
    p = predicates(_edited(c, u=axes, omega=speeds))
    assert bool(p.opp_spin[0, 0, 1])
    assert not bool(p.opp_spin[0, 0, 0])


def test_phaselock_true_for_shared_axis_and_phase(generator: torch.Generator) -> None:
    c = make_constellation(1, 16, 4, 2, generator, interior=True)
    axes = c.u.clone()
    axes[0, 1] = axes[0, 0]
    # force the phases onto exact values so the wrap seam cannot sit between them.
    phases = torch.zeros_like(c.phi)
    phases[0, 2] = 3.0
    axes[0, 2] = axes[0, 0]
    p = predicates(_edited(c, u=axes, phi=phases))
    # same axis, same phase: locked. same axis, far phase: aligned but not locked.
    assert bool(p.phaselock[0, 0, 1])
    assert bool(p.aligned[0, 0, 2])
    assert not bool(p.phaselock[0, 0, 2])


def test_n_near_counts_four_placed_satellites(generator: torch.Generator) -> None:
    c = make_constellation(1, 16, 4, 2, generator, interior=False)
    # pin the controller so the near and far placements stay inside the box.
    center = torch.tensor([[0.5, 0.5, 0.5]], dtype=c.x_c.dtype)
    near = (
        torch.tensor(
            [[0.05, 0.0, 0.0], [0.10, 0.0, 0.0], [0.15, 0.0, 0.0], [0.20, 0.0, 0.0]],
            dtype=c.x.dtype,
        )
        + center
    )
    # the twelve far satellites sit on a far grid, every one above TAU_D from x_c.
    far = torch.tensor(
        [
            [0.15, 0.15, 0.15],
            [0.85, 0.15, 0.15],
            [0.15, 0.85, 0.15],
            [0.15, 0.15, 0.85],
            [0.85, 0.85, 0.15],
            [0.85, 0.15, 0.85],
            [0.15, 0.85, 0.85],
            [0.85, 0.85, 0.85],
            [0.15, 0.50, 0.50],
            [0.85, 0.50, 0.50],
            [0.50, 0.15, 0.50],
            [0.50, 0.85, 0.50],
        ],
        dtype=c.x.dtype,
    )
    dist = (far - center).norm(dim=-1)
    assert bool((dist >= 0.3).all()), "the twelve far satellites must sit above TAU_D"
    moved = torch.cat((near.unsqueeze(0), far.unsqueeze(0)), dim=1)
    p = predicates(_edited(c, x=moved, x_c=center))
    assert int(p.n_near[0]) == 4


@given(st.integers(min_value=1, max_value=3))
def test_predicates_diagonal_false_and_counts_match_sums(seed_offset: int) -> None:
    gen = torch.Generator().manual_seed(SEED + 881 + seed_offset)
    c = make_constellation(2, 16, 4, 2, gen, interior=False)
    p = predicates(c)
    for pair in (p.aligned, p.phaselock, p.opp_spin):
        assert not bool(pair.diagonal(dim1=1, dim2=2).any())
    assert bool((p.n_near == p.near.sum(dim=-1)).all())
    assert bool((p.n_fast_near == (p.fast & p.near).sum(dim=-1)).all())
    assert bool((p.n_aligned_pairs == p.aligned.sum(dim=(-1, -2)) // 2).all())

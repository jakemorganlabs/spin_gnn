# tests for the synthetic samplers, the constructed Task B generator, and the
# Task B labeler. spec: docs/SPEC.md.
# flow:
# 1. _edited swaps one field on the frozen container.
# 2. _all_u and _all_x place every axis or radius on a hand-set value.
# 3. examples cover the sampler, the four labeler branches, the constructor,
#    and the uniform class frequencies.
# 4. properties cover rotation invariance of the label and exact constructor
#    labels.

import torch
from hypothesis import given
from hypothesis import strategies as st

from spin_gnn.constants import NEAR_R_MIN, R_INTERIOR, SEED
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
    class_frequencies,
    construct_task_b,
    construct_task_b_flat,
    label_task_b,
    sample_interior,
    sample_uniform,
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


def test_interior_sampler_batch_of_8(generator: torch.Generator) -> None:
    c = sample_interior(8, SpinGnnConfig(), generator)
    assert isinstance(validate(c), ValidOutcome)
    assert bool(in_interior(c).all())
    assert bool((c.x_c == 0.5).all())
    radii = (c.x - c.x_c.unsqueeze(1)).norm(dim=-1)
    assert bool(((radii >= NEAR_R_MIN) & (radii <= R_INTERIOR)).all())
    assert bool((c.h == 0).all())
    assert bool((c.h_c == 0).all())
    assert bool((c.v == 0).all())


def test_labeler_two_near_is_class_0(generator: torch.Generator) -> None:
    # two near and the rest just inside the interior gives n_near 2, class 0.
    c = sample_interior(1, SpinGnnConfig(), generator, dtype=torch.float64)
    radii = torch.full((1, c.x.shape[1]), 0.3, dtype=c.x.dtype)
    radii[0, :2] = 0.1
    direction = torch.randn(1, c.x.shape[1], 3, generator=generator, dtype=c.x.dtype)
    direction = unit_axis_exact(direction)
    x = c.x_c.unsqueeze(1) + radii.unsqueeze(-1) * direction
    out = label_task_b(_edited(c, x=x))
    assert out.tolist() == [0]


def test_labeler_four_near_slow_is_class_1(generator: torch.Generator) -> None:
    # many near with every speed slow lands on class 1, even with the near
    # satellites one diameter apart so no pair of them can align.
    config = SpinGnnConfig()
    c = sample_interior(1, config, generator, dtype=torch.float64)
    n = c.x.shape[1]
    k = max(4, n // 2)
    radii = torch.full((1, n), 0.3, dtype=c.x.dtype)
    radii[0, :k] = 0.1
    direction = torch.randn(1, n, 3, generator=generator, dtype=c.x.dtype)
    direction = unit_axis_exact(direction)
    x = c.x_c.unsqueeze(1) + radii.unsqueeze(-1) * direction
    omega = torch.full((1, n), 0.1, dtype=c.omega.dtype)
    out = label_task_b(_edited(c, x=x, omega=omega))
    assert out.tolist() == [1]


def test_labeler_four_near_fast_aligned_is_class_2(generator: torch.Generator) -> None:
    c = sample_interior(1, SpinGnnConfig(), generator, dtype=torch.float64)
    n = c.x.shape[1]
    k = max(4, n // 2)
    radii = torch.full((1, n), 0.3, dtype=c.x.dtype)
    radii[0, :k] = 0.1
    direction = torch.randn(1, n, 3, generator=generator, dtype=c.x.dtype)
    direction = unit_axis_exact(direction)
    x = c.x_c.unsqueeze(1) + radii.unsqueeze(-1) * direction
    omega = torch.full((1, n), 1.5, dtype=c.omega.dtype)
    u = c.u.clone()
    u[0, 1] = u[0, 0]
    out = label_task_b(_edited(c, x=x, omega=omega, u=u))
    assert out.tolist() == [2]


def test_labeler_four_near_fast_unaligned_is_class_3(generator: torch.Generator) -> None:
    # a greedy set of axes, every pair under cosine 0.9, so no pair anywhere
    # in the constellation aligns at the 0.95 predicate.
    config = SpinGnnConfig()
    c = sample_interior(1, config, generator, dtype=torch.float64)
    n = c.x.shape[1]
    k = max(4, n // 2)
    radii = torch.full((1, n), 0.3, dtype=c.x.dtype)
    radii[0, :k] = 0.1
    direction = torch.randn(1, n, 3, generator=generator, dtype=c.x.dtype)
    direction = unit_axis_exact(direction)
    x = c.x_c.unsqueeze(1) + radii.unsqueeze(-1) * direction
    omega = torch.full((1, n), 1.5, dtype=c.omega.dtype)
    accepted: list[torch.Tensor] = []
    candidate_count = 0
    while len(accepted) < n and candidate_count < 20000:
        candidate = unit_axis_exact(
            torch.randn(1, 3, generator=generator, dtype=c.u.dtype)
        )[0]
        candidate_count += 1
        if all(
            abs(float((candidate * prev).sum())) < 0.9 for prev in accepted
        ):
            accepted.append(candidate)
    assert len(accepted) == n, "greedy axes must pack n within the budget"
    u = torch.stack(accepted).unsqueeze(0)
    out = label_task_b(_edited(c, x=x, omega=omega, u=u))
    assert out.tolist() == [3]


def test_constructor_hits_every_class_1000_of_1000() -> None:
    config = SpinGnnConfig()
    for class_id in range(4):
        generator = torch.Generator().manual_seed(SEED + class_id)
        c = construct_task_b(1000, class_id, config, generator)
        labels = label_task_b(c)
        assert int((labels == class_id).sum()) == 1000


def test_constructor_flat_64_splits_evenly(generator: torch.Generator) -> None:
    config = SpinGnnConfig()
    c, labels = construct_task_b_flat(64, config, generator)
    assert bool((torch.bincount(labels, minlength=4) == 16).all())
    assert bool((label_task_b(c) == labels).all())


def test_uniform_frequencies_sum_to_one(generator: torch.Generator) -> None:
    # the uniform draw collapses the class mix; print the four frequencies.
    config = SpinGnnConfig()
    labels = label_task_b(sample_uniform(4000, config, generator))
    freq = class_frequencies(labels)
    print("\nuniform class frequencies:", [round(float(v), 6) for v in freq])
    assert bool((freq.sum() - 1.0).abs() <= 1e-6)


def test_constructor_includes_class_and_attempt_in_failure(
    generator: torch.Generator,
) -> None:
    try:
        construct_task_b(1, 7, SpinGnnConfig(), generator)
    except AssertionError as err:
        assert "7" in str(err)
    else:
        raise AssertionError("construct_task_b must reject a class outside 0..3")


@given(st.integers(min_value=1, max_value=6))
def test_label_invariant_under_haar_rotation(seed_offset: int) -> None:
    # Property 1. the label reads only distances from x_c and axis cosines,
    # both fixed by a rotation about the controller. the batches mix
    # constructed classes so every branch of the rule faces a rotation.
    generator = torch.Generator().manual_seed(SEED + 41 + seed_offset)
    config = SpinGnnConfig()
    pieces = [
        construct_task_b(2, class_id, config, generator, dtype=torch.float64)
        for class_id in range(4)
    ]
    mixed = Constellation(
        x=torch.cat([p.x for p in pieces]),
        s=torch.cat([p.s for p in pieces]),
        u=torch.cat([p.u for p in pieces]),
        phi=torch.cat([p.phi for p in pieces]),
        omega=torch.cat([p.omega for p in pieces]),
        h=torch.cat([p.h for p in pieces]),
        v=torch.cat([p.v for p in pieces]),
        x_c=torch.cat([p.x_c for p in pieces]),
        h_c=torch.cat([p.h_c for p in pieces]),
    )
    probe = sample_interior(4, config, generator, dtype=torch.float64)
    batch = Constellation(
        x=torch.cat([mixed.x, probe.x]),
        s=torch.cat([mixed.s, probe.s]),
        u=torch.cat([mixed.u, probe.u]),
        phi=torch.cat([mixed.phi, probe.phi]),
        omega=torch.cat([mixed.omega, probe.omega]),
        h=torch.cat([mixed.h, probe.h]),
        v=torch.cat([mixed.v, probe.v]),
        x_c=torch.cat([mixed.x_c, probe.x_c]),
        h_c=torch.cat([mixed.h_c, probe.h_c]),
    )
    assert_valid(batch)
    rotation = random_rotation(batch.x.shape[0], generator)
    rotated = rotate_about_controller(batch, rotation)
    assert bool(in_interior(rotated).all())
    before = label_task_b(batch)
    after = label_task_b(rotated)
    assert bool((before == after).all())


@given(st.integers(min_value=0, max_value=3), st.integers(min_value=1, max_value=4))
def test_construct_matches_class(class_id: int, seed_offset: int) -> None:
    # Property 3. small constructed batches label as the class asked.
    generator = torch.Generator().manual_seed(SEED + 97 + 8 * seed_offset + class_id)
    config = SpinGnnConfig()
    c = construct_task_b(8, class_id, config, generator)
    labels = label_task_b(c)
    assert bool((labels == class_id).all())
    assert bool(in_interior(c).all())

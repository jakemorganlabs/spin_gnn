# examples and properties for box.py.
# flow:
# 1. pin the documented examples as sentence tests.
# 2. pin the reflection, projection, wall feature, box coords, and pairwise properties.

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st

from spin_gnn.constants import BOX
from spin_gnn.geometry.box import (
    box_coords,
    pairwise_displacement,
    pairwise_distance,
    project_to_box,
    reflect_into_box,
    wall_features,
)
from spin_gnn.tests.conftest import box_points, displacements, free_points

F64 = torch.float64


# examples: fixed input, fixed output, read as sentences.


def test_project_to_box_clamps_outside_corners() -> None:
    x = torch.tensor([1.5, 0.5, -0.2], dtype=F64)
    out = project_to_box(x)
    assert torch.allclose(out, torch.tensor([1.0, 0.5, 0.0], dtype=F64))


def test_project_to_box_leaves_interior_point_unchanged() -> None:
    x = torch.tensor([0.0, 1.0, 0.5], dtype=F64)
    # boundary points on the near wall stay put.
    assert torch.equal(project_to_box(x), x)


def test_reflect_into_box_bounces_off_far_wall() -> None:
    x = torch.tensor([0.9, 0.5, 0.5], dtype=F64)
    delta = torch.tensor([0.3, 0.0, 0.0], dtype=F64)
    x_out, d_out = reflect_into_box(x, delta)
    assert torch.allclose(x_out, torch.tensor([0.8, 0.5, 0.5], dtype=F64), atol=1e-9)
    assert torch.allclose(d_out, torch.tensor([-0.3, 0.0, 0.0], dtype=F64), atol=1e-9)


def test_reflect_into_box_passes_through_when_inside() -> None:
    x = torch.tensor([0.5, 0.5, 0.5], dtype=F64)
    delta = torch.tensor([0.1, 0.1, 0.1], dtype=F64)
    x_out, d_out = reflect_into_box(x, delta)
    assert torch.allclose(x_out, torch.tensor([0.6, 0.6, 0.6], dtype=F64), atol=1e-9)
    assert torch.allclose(d_out, delta, atol=1e-9)


def test_wall_features_at_center_are_all_half_extent() -> None:
    x = torch.tensor([0.5, 0.5, 0.5], dtype=F64)
    feats = wall_features(x)
    assert torch.allclose(feats, torch.full((6,), 0.5, dtype=F64), atol=1e-9)


def test_box_coords_at_center() -> None:
    x = torch.tensor([0.5, 0.5, 0.5], dtype=F64)
    coords = box_coords(x)
    assert torch.allclose(coords, torch.tensor([0.5, 0.5, 0.5, 0.25, 0.25, 0.25], dtype=F64))


def test_pairwise_distance_between_opposite_corners() -> None:
    a = torch.tensor([0.0, 0.0, 0.0], dtype=F64)
    b = torch.tensor([1.0, 1.0, 1.0], dtype=F64)
    x = torch.stack((a, b)).unsqueeze(0)
    r = pairwise_displacement(x)
    d = pairwise_distance(r)
    assert torch.allclose(d[0, 0, 1], torch.sqrt(torch.tensor(3.0, dtype=F64)), atol=1e-9)


# properties: Hypothesis, float64, deterministic and deadline-free.


@given(x=box_points(), delta=displacements(1.9))
def test_reflect_into_box_stays_inside(x: np.ndarray, delta: np.ndarray) -> None:
    x_t = torch.from_numpy(np.ascontiguousarray(x)).to(F64)
    d_t = torch.from_numpy(np.ascontiguousarray(delta)).to(F64)
    x_out, _ = reflect_into_box(x_t, d_t)
    lo = torch.zeros(3, dtype=F64)
    hi = torch.tensor(BOX, dtype=F64)
    assert bool(((x_out >= lo) & (x_out <= hi)).all())


@given(x=box_points(), delta=displacements(1.9))
def test_reflect_into_box_preserves_displacement_magnitude(
    x: np.ndarray, delta: np.ndarray
) -> None:
    x_t = torch.from_numpy(np.ascontiguousarray(x)).to(F64)
    d_t = torch.from_numpy(np.ascontiguousarray(delta)).to(F64)
    _, d_out = reflect_into_box(x_t, d_t)
    assert bool(((d_out.abs() - d_t.abs()).abs() <= 1e-9).all())


@given(x=free_points(low=-3.0, high=3.0))
def test_project_to_box_is_idempotent_and_inside(x: np.ndarray) -> None:
    x_t = torch.from_numpy(np.ascontiguousarray(x)).to(F64)
    once = project_to_box(x_t)
    twice = project_to_box(once)
    lo = torch.zeros(3, dtype=F64)
    hi = torch.tensor(BOX, dtype=F64)
    assert bool(((once >= lo) & (once <= hi)).all())
    assert torch.equal(once, twice)


@given(x=box_points())
def test_wall_features_are_nonnegative_and_sum_to_extent(x: np.ndarray) -> None:
    x_t = torch.from_numpy(np.ascontiguousarray(x)).to(F64)
    feats = wall_features(x_t)
    assert bool((feats >= 0).all())
    pairs = feats.reshape(3, 2)
    sums = pairs.sum(dim=-1)
    extent = torch.tensor(BOX, dtype=F64)
    assert torch.allclose(sums, extent, atol=1e-9)


@given(seed=st.integers(0, 2**31 - 1))
def test_pairwise_displacement_is_antisymmetric_with_zero_diagonal(seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(2, 5, 3, generator=g, dtype=F64)
    r = pairwise_displacement(x)
    assert bool((r + r.transpose(1, 2) == 0).all())
    diag = r.diagonal(dim1=1, dim2=2)
    assert bool((diag == 0).all())

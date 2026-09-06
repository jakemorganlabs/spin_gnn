# examples and properties for frames.py.
# flow:
# 1. pin the octahedral group example.
# 2. pin the Haar rotation and rotation-application properties.
# 3. pin a seeded statistical check on the Haar mean.

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st

from spin_gnn.geometry.box import assert_vec3, safe_norm
from spin_gnn.geometry.frames import (
    octahedral_rotations,
    random_rotation,
    rotate_channels,
    rotate_vectors,
)
from spin_gnn.geometry.spin import axis_alignment
from spin_gnn.tests.conftest import unit_vectors

F64 = torch.float64


# example: the octahedral group is exactly 24 rotations with the identity present.


def test_octahedral_rotations_are_24_with_identity() -> None:
    rots = octahedral_rotations()
    assert rots.shape == (24, 3, 3)
    identity = torch.eye(3, dtype=F64)
    assert bool(((rots - identity).abs().amax(dim=(-1, -2)) <= 1e-12).any())
    # every entry is a sign or a zero and every determinant is +1.
    assert bool(((rots.abs() == 0) | (rots.abs() == 1)).all())
    dets = torch.linalg.det(rots)
    assert torch.allclose(dets, torch.ones(24, dtype=F64), atol=1e-12)


def test_rotate_vectors_applies_rotation_matrix() -> None:
    # rotating e_x by +90 degrees about z lands on e_y.
    c, s = 0.0, 1.0
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=F64).unsqueeze(0)
    v = torch.tensor([[1.0, 0.0, 0.0]], dtype=F64)
    out = rotate_vectors(rot, v)
    assert torch.allclose(out, torch.tensor([[0.0, 1.0, 0.0]], dtype=F64), atol=1e-9)


# properties: Haar sample shape, det, and closure of the octahedral group.


@given(n=st.integers(1, 8), seed=st.integers(0, 1000))
def test_random_rotation_is_orthonormal_with_det_one(n: int, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    rot = random_rotation(n, g)
    assert rot.shape == (n, 3, 3)
    eye = torch.eye(3, dtype=F64).expand(n, 3, 3)
    assert torch.allclose(rot.transpose(-1, -2) @ rot, eye, atol=1e-9, rtol=0)
    dets = torch.linalg.det(rot)
    assert torch.allclose(dets, torch.ones(n, dtype=F64), atol=1e-9, rtol=0)


@given(a=st.integers(0, 23), b=st.integers(0, 23))
def test_octahedral_group_is_closed_under_product(a: int, b: int) -> None:
    rots = octahedral_rotations()
    product = rots[a] @ rots[b]
    gaps = (rots - product).abs().amax(dim=(-1, -2))
    assert bool((gaps <= 1e-12).any())


@given(u=unit_vectors(), seed=st.integers(0, 1000))
def test_rotation_preserves_alignment_and_norm(u: np.ndarray, seed: int) -> None:
    # snap to a face or corner axis so the unit vector is exact in float64.
    u_snap = np.asarray(u, dtype=np.float64)
    idx = int(np.argmax(np.abs(u_snap)))
    snapped = np.zeros(3, dtype=np.float64)
    snapped[idx] = float(np.sign(u_snap[idx]))
    u_t = torch.from_numpy(snapped).to(F64)
    # property 11: a rotated unit vector aligns with itself and stays unit.
    g = torch.Generator().manual_seed(seed)
    rot = random_rotation(1, g)[0]
    ru = rot @ u_t
    assert torch.allclose(axis_alignment(ru, ru), torch.tensor(1.0, dtype=F64), atol=1e-8)
    batched_u = u_t.unsqueeze(0)
    batched_rot = rot.unsqueeze(0)
    rotated = rotate_vectors(batched_rot, batched_u).squeeze(0)
    assert torch.allclose(safe_norm(rotated), torch.tensor(1.0, dtype=F64), atol=1e-9)


@given(u=unit_vectors(), seed=st.integers(0, 1000))
def test_rotate_channels_preserves_channel_norms(u: np.ndarray, seed: int) -> None:
    u_t = torch.from_numpy(np.ascontiguousarray(u)).to(F64)
    g = torch.Generator().manual_seed(seed)
    rot = random_rotation(1, g)
    # lay the unit vector across c channels of a (B, N, 3, c) tensor.
    v = u_t.reshape(1, 1, 3, 1).expand(1, 2, 3, 4).contiguous()
    out = rotate_channels(rot, v)
    # each channel holds a 3-vector, so the norm lives on axis -2.
    before = torch.sqrt((v * v).sum(-2))
    after = torch.sqrt((out * out).sum(-2))
    assert torch.allclose(after, before, atol=1e-9, rtol=0)
    assert_vec3(out[:, :, :, 0], "rotate_channels output")


# statistical check: the Haar sample mean is near zero, which catches a sign fix bug.


def test_haar_mean_entries_are_near_zero() -> None:
    n_haar = 10000
    g = torch.Generator().manual_seed(0)
    rots = random_rotation(n_haar, g)
    mean = rots.mean(dim=0)
    assert bool((mean.abs() < 0.05).all())

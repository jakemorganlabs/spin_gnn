"""Prop 2 (message equivariance)."""

# flow:
# 1. examples pin shapes, the zero diagonal, and an exact distance.
# 2. the property proves scalar outputs invariant and the vector message
#    equivariant under Haar rotations on the interior.

import torch
from hypothesis import given
from hypothesis import strategies as st

from spin_gnn.constants import SEED
from spin_gnn.constellation import rotate_about_controller
from spin_gnn.geometry.frames import random_rotation, rotate_channels
from spin_gnn.layers.message import MessageLayer
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.tests.test_constellation import make_constellation
from spin_gnn.types import Constellation


def _seeded_layer() -> MessageLayer:
    torch.manual_seed(SEED)
    return MessageLayer(SpinGnnConfig(n=6, d=16, c=4, d_c=32, d_m=16)).double()


def _with_x_h(c: Constellation, x: torch.Tensor, h: torch.Tensor) -> Constellation:
    return Constellation(
        x=x, s=c.s, u=c.u, phi=c.phi, omega=c.omega, h=h, v=c.v, x_c=c.x_c, h_c=c.h_c
    )


def test_message_shapes_and_zero_diagonal(generator: torch.Generator) -> None:
    layer = _seeded_layer()
    c = make_constellation(3, 6, 16, 4, generator, interior=True)
    msg = layer(c)
    assert msg.m_edge.shape == (3, 6, 6, 16)
    assert msg.gate_x.shape == (3, 6, 6)
    assert msg.m_i.shape == (3, 6, 16)
    assert msg.big_m_i.shape == (3, 6, 3, 4)
    assert msg.m_c.shape == (3, 16)
    assert msg.r_hat.shape == (3, 6, 6, 3)
    assert msg.dist.shape == (3, 6, 6)
    # the diagonal is not an edge, so both scalar products are exactly zero there.
    assert bool((msg.m_edge.diagonal(dim1=1, dim2=2) == 0).all())
    assert bool((msg.gate_x.diagonal(dim1=1, dim2=2) == 0).all())


def test_message_reports_exact_pair_distance(generator: torch.Generator) -> None:
    layer = _seeded_layer()
    c = make_constellation(1, 6, 16, 4, generator, interior=True)
    moved = c.x.clone()
    moved[0, 0] = torch.tensor([0.5, 0.5, 0.5], dtype=c.x.dtype)
    moved[0, 1] = torch.tensor([0.8, 0.5, 0.5], dtype=c.x.dtype)
    msg = layer(_with_x_h(c, moved, c.h))
    assert abs(float(msg.dist[0, 0, 1]) - 0.3) <= 1e-9
    assert abs(float(msg.dist[0, 1, 0]) - 0.3) <= 1e-9


def test_message_controller_output_with_zero_h(generator: torch.Generator) -> None:
    layer = _seeded_layer()
    c = make_constellation(3, 6, 16, 4, generator, interior=True)
    zero_h = torch.zeros_like(c.h)
    msg = layer(_with_x_h(c, c.x, zero_h))
    assert msg.m_c.shape == (3, 16)
    assert bool(torch.isfinite(msg.m_c).all())


@given(st.integers(min_value=1, max_value=3))
def test_message_equivariance(seed_offset: int) -> None:
    # Prop 2: scalar outputs are invariant, big_m_i is equivariant, under rho_R.
    gen = torch.Generator().manual_seed(SEED + 97 + seed_offset)
    layer = _seeded_layer()
    c = make_constellation(3, 6, 16, 4, gen, interior=True)
    rotation = random_rotation(3, gen)
    before = layer(c)
    after = layer(rotate_about_controller(c, rotation))
    for name in ("m_i", "m_c", "m_edge", "gate_x"):
        gap = (getattr(before, name) - getattr(after, name)).abs()
        assert bool((gap <= 1e-9).all()), f"{name} moved by {float(gap.max())}"
    rotated_big_m = rotate_channels(rotation, before.big_m_i)
    gap = (rotated_big_m - after.big_m_i).abs()
    assert bool((gap <= 1e-9).all()), f"big_m_i moved by {float(gap.max())}"

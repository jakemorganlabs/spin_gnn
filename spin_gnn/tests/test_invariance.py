"""Prop 1 (packet invariance). Session 4 adds Prop 3 and the guard test."""

# flow:
# 1. packet examples pin the column values and the width formula.
# 2. properties prove Prop 1 for the edge packet and the self packet.
# 3. the swap property pins the column behavior exactly.

import torch
from hypothesis import given
from hypothesis import strategies as st

from spin_gnn.constants import MARGIN, R_INTERIOR, SEED
from spin_gnn.constellation import rotate_about_controller
from spin_gnn.geometry.frames import random_rotation
from spin_gnn.geometry.invariants import (
    PacketColumn,
    edge_packet,
    packet_width,
    self_packet,
    size_pair,
)
from spin_gnn.tests.test_constellation import make_constellation


def test_packet_width_counts_scalar_blocks() -> None:
    assert packet_width(64) == 142


def test_r_interior_matches_margin_definition() -> None:
    assert abs(R_INTERIOR - (0.5 - MARGIN)) <= 1e-12


def test_edge_packet_reports_exact_distance(generator: torch.Generator) -> None:
    c = make_constellation(1, 8, 4, 2, generator, interior=True)
    moved = c.x.clone()
    moved[0, 0] = torch.tensor([0.5, 0.5, 0.5], dtype=c.x.dtype)
    moved[0, 1] = torch.tensor([0.8, 0.5, 0.5], dtype=c.x.dtype)
    packet = edge_packet(_with_x(c, moved))
    assert abs(float(packet[0, 0, 1, PacketColumn.D]) - 0.3) <= 1e-9
    assert abs(float(packet[0, 1, 0, PacketColumn.D]) - 0.3) <= 1e-9


def test_edge_packet_alpha_is_one_for_aligned_axes(generator: torch.Generator) -> None:
    c = make_constellation(1, 8, 4, 2, generator, interior=True)
    aligned = c.u.clone()
    aligned[0, 1] = aligned[0, 0]
    packet = edge_packet(_with_u(c, aligned))
    assert abs(float(packet[0, 0, 1, PacketColumn.ALPHA]) - 1.0) <= 1e-9


def test_size_pair_matches_log_ratio_and_relative_size() -> None:
    s_i = torch.tensor(2.0, dtype=torch.float64)
    s_j = torch.tensor(1.0, dtype=torch.float64)
    rho, sigma = size_pair(s_i, s_j)
    assert abs(float(rho) - float(torch.log(s_i / s_j))) <= 1e-9
    assert abs(float(sigma) - 2.0 / 3.0) <= 1e-9


def test_self_packet_convention_columns(generator: torch.Generator) -> None:
    c = make_constellation(1, 8, 4, 2, generator, interior=True)
    h_c_proj = torch.zeros(1, 4, dtype=c.h.dtype)
    packet = self_packet(c, h_c_proj)
    assert bool((packet[..., PacketColumn.COS_PSI_C] == 1).all())
    # the controller axis lies along its own sight line, so the column is 1.
    assert bool((packet[..., PacketColumn.U_J_TO_C] == 1).all())
    # beta copies alpha under the axis convention.
    assert bool(
        (packet[..., PacketColumn.ALPHA] - packet[..., PacketColumn.BETA_I_J]).abs().max() <= 1e-12
    )


def _with_x(c, x):
    return type(c)(x=x, s=c.s, u=c.u, phi=c.phi, omega=c.omega, h=c.h, v=c.v, x_c=c.x_c, h_c=c.h_c)


def _with_u(c, u):
    return type(c)(x=c.x, s=c.s, u=u, phi=c.phi, omega=c.omega, h=c.h, v=c.v, x_c=c.x_c, h_c=c.h_c)


@given(st.integers(min_value=1, max_value=3))
def test_packet_invariant(seed_offset: int) -> None:
    # Prop 1: every column of the edge packet is invariant under rho_R.
    gen = torch.Generator().manual_seed(SEED + 51 + seed_offset)
    c = make_constellation(2, 8, 4, 2, gen, interior=True)
    rotation = random_rotation(2, gen)
    before = edge_packet(c)
    after = edge_packet(rotate_about_controller(c, rotation))
    gap = (before - after).abs()
    assert bool((gap <= 1e-9).all()), f"packet moved by {float(gap.max())}"


@given(st.integers(min_value=1, max_value=3))
def test_self_packet_invariant(seed_offset: int) -> None:
    gen = torch.Generator().manual_seed(SEED + 171 + seed_offset)
    c = make_constellation(2, 8, 4, 2, gen, interior=True)
    h_c_proj = torch.randn(2, 4, generator=gen, dtype=torch.float64)
    rotation = random_rotation(2, gen)
    before = self_packet(c, h_c_proj)
    after = self_packet(rotate_about_controller(c, rotation), h_c_proj)
    gap = (before - after).abs()
    assert bool((gap <= 1e-9).all()), f"self packet moved by {float(gap.max())}"


@given(st.integers(min_value=1, max_value=2))
def test_edge_packet_swap_contract(seed_offset: int) -> None:
    gen = torch.Generator().manual_seed(SEED + 731 + seed_offset)
    c = make_constellation(1, 8, 4, 2, gen, interior=True)
    packet = edge_packet(c)
    swapped = packet.transpose(1, 2)
    d = int(c.h.shape[-1])
    # every named column matches the swap contract; float columns sit at half
    # a dozen ulps because the two directions are computed independently.
    assert bool(
        (swapped[..., PacketColumn.RHO] + packet[..., PacketColumn.RHO]).abs().max() <= 1e-12
    )
    assert bool(
        (swapped[..., PacketColumn.DOMEGA] + packet[..., PacketColumn.DOMEGA]).abs().max()
        <= 1e-12
    )
    assert bool(
        (swapped[..., PacketColumn.SIN_DPHI] + packet[..., PacketColumn.SIN_DPHI])
        .abs()
        .max()
        <= 1e-12
    )
    assert bool(
        (swapped[..., PacketColumn.BETA_I_J] - packet[..., PacketColumn.BETA_J_I])
        .abs()
        .max()
        <= 1e-12
    )
    assert bool(
        (swapped[..., PacketColumn.U_I_TO_C] - packet[..., PacketColumn.U_J_TO_C])
        .abs()
        .max()
        <= 1e-12
    )
    h_i_start = PacketColumn.H_I_START
    h_j_start = h_i_start + d
    assert bool(
        (
            swapped[..., h_i_start:h_j_start] - packet[..., h_j_start : h_j_start + d] == 0
        ).all()
    )
    # symmetric columns stay fixed under the swap.
    for column in (
        PacketColumn.D,
        PacketColumn.LOG_D,
        PacketColumn.ALPHA,
        PacketColumn.COS_DPHI,
        PacketColumn.OMEGA_PROD,
        PacketColumn.COS_PSI_C,
    ):
        assert bool(
            (swapped[..., column] - packet[..., column]).abs().max() <= 1e-12
        ), f"column {column.name} must be swap-symmetric"

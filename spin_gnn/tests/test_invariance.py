"""Prop 1 and Prop 3, plus the guard."""

# flow:
# 1. packet examples pin the column values and the width formula.
# 2. properties prove Prop 1 for the edge packet and the self packet.
# 3. the swap property pins the column behavior exactly.
# 4. the model-level Prop 3 invariant-outputs property and the guard live here:
#    the guard shows that leaking box coordinates into h breaks the invariant.

import torch
from hypothesis import given
from hypothesis import strategies as st
from torch import Tensor

from spin_gnn.constants import MARGIN, R_INTERIOR, SEED
from spin_gnn.constellation import rotate_about_controller
from spin_gnn.geometry.box import box_coords
from spin_gnn.geometry.frames import random_rotation
from spin_gnn.geometry.invariants import (
    PacketColumn,
    edge_packet,
    packet_width,
    self_packet,
    size_pair,
)
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.model.spin_gnn_gnn import build_model
from spin_gnn.tests.test_constellation import make_constellation
from spin_gnn.types import Constellation


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


# session 4: Prop 3 at model level, and the guard that shows where it breaks.


def _small_config() -> SpinGnnConfig:
    return SpinGnnConfig(n=8, d=16, c=4, d_c=32, d_m=16, n_layers=2)


@given(st.integers(min_value=1, max_value=3))
def test_model_invariant_outputs_under_haar(seed_offset: int) -> None:
    # Prop 3: h_c, y, and z are invariant under Haar rotations on the interior.
    config = _small_config()
    model = build_model(config).double()
    gen = torch.Generator().manual_seed(SEED + 553 + seed_offset)
    c = make_constellation(2, config.n, config.d, config.c, gen, interior=True)
    rotation = random_rotation(2, gen)
    before = model(c)
    after = model(rotate_about_controller(c, rotation))
    for name, tensor_a, tensor_b in (
        ("h_c", before.constellation.h_c, after.constellation.h_c),
        ("y", before.y, after.y),
        ("z", before.z, after.z),
    ):
        gap = (tensor_a - tensor_b).abs()
        assert bool((gap <= 1e-8).all()), f"{name} moved by {float(gap.max())}"


def _inject_box_coords(
    model: torch.nn.Module, c: Constellation, leak: torch.nn.Linear
) -> Tensor:
    # step 1: encode, then leak box coordinates into h before the layers run.
    state = model.encoder(c)  # type: ignore[attr-defined]
    leaked = Constellation(
        x=state.x,
        s=state.s,
        u=state.u,
        phi=state.phi,
        omega=state.omega,
        h=state.h + leak(box_coords(state.x)),
        v=state.v,
        x_c=state.x_c,
        h_c=state.h_c,
    )
    # step 2: run the layers on the leaked state and return the final h_c.
    for layer in model.layers:  # type: ignore[attr-defined]
        leaked = layer(leaked)
    return leaked.h_c


def test_box_coordinate_injection_breaks_invariance(generator: torch.Generator) -> None:
    # the guard: box_coords does not co-rotate, so leaking it into h through a
    # fixed Linear must move h_c by more than 1e-3 under a generic rotation.
    config = _small_config()
    model = build_model(config).double()
    torch.manual_seed(0)
    leak = torch.nn.Linear(6, config.d, dtype=torch.float64)
    with torch.no_grad():
        leak.weight.fill_(1.0)
        leak.bias.fill_(0.0)
    c = make_constellation(2, config.n, config.d, config.c, generator, interior=True)
    rotation = random_rotation(2, generator)
    h_c_before = _inject_box_coords(model, c, leak)
    h_c_after = _inject_box_coords(model, rotate_about_controller(c, rotation), leak)
    gap = (h_c_before - h_c_after).abs().max().detach()
    print(f"\nguard gap: {float(gap)}")
    assert float(gap) > 1e-3

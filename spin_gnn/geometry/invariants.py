# the SO(3)-invariant edge packet and the controller self packet, per Prop 1.
# every column is an inner product of two co-rotating vectors, a norm of such
# a vector, or a scalar that does not transform; so the packet is invariant.
# flow:
# 1. declare the column enum and the packet width.
# 2. declare the pair helpers: controller angle and size pair.
# 3. build the full N by N edge packet by broadcasting, one cat at the end.
# 4. build the controller self packet under the section 3 convention.
# the packet reads distances, inner products, scalars, and h. it never reads
# wall features, box coordinates, or any raw coordinate.

from enum import IntEnum

import torch
from torch import Tensor

from spin_gnn.constants import EPS, OMEGA_MAX
from spin_gnn.constellation import assert_valid
from spin_gnn.geometry.spin import axis_alignment, phase_difference, spine_to_line
from spin_gnn.types import Constellation


class PacketColumn(IntEnum):
    # the exact column order of every packet row.
    D = 0
    LOG_D = 1
    RHO = 2
    SIGMA = 3
    ALPHA = 4
    SIN_DPHI = 5
    COS_DPHI = 6
    DOMEGA = 7
    OMEGA_PROD = 8
    BETA_I_J = 9
    BETA_J_I = 10
    COS_PSI_C = 11
    U_I_TO_C = 12
    U_J_TO_C = 13
    H_I_START = 14
    # h_j starts at 14 + d


N_SCALAR_COLUMNS: int = 14


def packet_width(d: int) -> int:
    assert d >= 1, "d must be at least 1"
    return N_SCALAR_COLUMNS + 2 * d


def controller_angle(x_i: Tensor, x_j: Tensor, x_c: Tensor) -> Tensor:
    # step 1: sight lines from the controller to each satellite.
    v_i = x_i - x_c
    v_j = x_j - x_c
    cos_raw = (v_i * v_j).sum(dim=-1) / (v_i.norm(dim=-1) * v_j.norm(dim=-1) + EPS)
    cos_psi = torch.clamp(cos_raw, -1.0, 1.0)
    # step 2: the cosine is symmetric in (i, j) and lies in [-1, 1].
    v_ji = x_j - x_c
    swapped = (v_ji * (x_i - x_c)).sum(dim=-1) / (
        v_ji.norm(dim=-1) * (x_i - x_c).norm(dim=-1) + EPS
    )
    assert bool(((cos_psi - torch.clamp(swapped, -1.0, 1.0)).abs() <= 1e-12).all()), (
        "controller_angle must be symmetric in (i, j)"
    )
    assert bool(((cos_psi >= -1 - 1e-9) & (cos_psi <= 1 + 1e-9)).all())
    return cos_psi


def size_pair(s_i: Tensor, s_j: Tensor) -> tuple[Tensor, Tensor]:
    assert s_i.shape == s_j.shape, (
        f"s_i and s_j must share shape, got {tuple(s_i.shape)} and {tuple(s_j.shape)}"
    )
    assert bool((s_i > 0).all()) and bool((s_j > 0).all()), "sizes must be positive"
    # step 1: rho is the log ratio, antisymmetric by construction.
    rho = torch.log(s_i / s_j)
    # step 2: sigma is the relative size, and the two directions sum to 1.
    sigma = s_i / (s_i + s_j)
    return rho, sigma


def edge_packet(c: Constellation) -> Tensor:
    # step 1: require a valid constellation before any geometry.
    assert_valid(c)
    b, n, d = int(c.x.shape[0]), int(c.x.shape[1]), int(c.h.shape[-1])

    # step 2: broadcast every field into ordered pairs indexed by (i, j).
    # r_ij = x_i - x_j per SPEC; d is the plain norm so the D column is exact,
    # and eps enters through LOG_D and the r_hat denominator only.
    r = c.x.unsqueeze(2) - c.x.unsqueeze(1)
    dist = r.norm(dim=-1)
    r_hat = r / dist.clamp(min=EPS).unsqueeze(-1)
    u_i = c.u.unsqueeze(2).expand(b, n, n, 3)
    u_j = c.u.unsqueeze(1).expand(b, n, n, 3)
    s_i = c.s.unsqueeze(2).expand(b, n, n)
    s_j = c.s.unsqueeze(1).expand(b, n, n)
    phi_i = c.phi.unsqueeze(2).expand(b, n, n)
    phi_j = c.phi.unsqueeze(1).expand(b, n, n)
    omega_i = c.omega.unsqueeze(2).expand(b, n, n)
    omega_j = c.omega.unsqueeze(1).expand(b, n, n)
    h_i = c.h.unsqueeze(2).expand(b, n, n, d)
    h_j = c.h.unsqueeze(1).expand(b, n, n, d)

    # step 3: compute the named columns as invariants. U_I_TO_C is the inner
    # product of the axis with the sight line to the controller: an inner
    # product of two co-rotating vectors, so Prop 1 covers it.
    rel_x_c = c.x - c.x_c.unsqueeze(1)
    dist_c = rel_x_c.norm(dim=-1)
    dir_i_to_c = -(rel_x_c / dist_c.clamp(min=EPS).unsqueeze(-1))
    axis_to_c = (c.u * dir_i_to_c).sum(dim=-1)
    rho, sigma = size_pair(s_i, s_j)
    alpha = axis_alignment(u_i, u_j)
    d_phi = phase_difference(phi_i, phi_j)
    d_omega = (omega_i - omega_j) / OMEGA_MAX
    omega_prod = omega_i * omega_j / (OMEGA_MAX * OMEGA_MAX)
    beta_i_j = spine_to_line(u_i, r_hat)
    beta_j_i = spine_to_line(u_j, -r_hat)
    cos_psi_c = controller_angle(
        c.x.unsqueeze(2).expand(b, n, n, 3),
        c.x.unsqueeze(1).expand(b, n, n, 3),
        c.x_c.unsqueeze(1).unsqueeze(1).expand(b, n, n, 3),
    )

    # step 4: stack in PacketColumn order and cat once.
    columns = [
        dist.unsqueeze(-1),
        torch.log(dist + EPS).unsqueeze(-1),
        rho.unsqueeze(-1),
        sigma.unsqueeze(-1),
        alpha.unsqueeze(-1),
        torch.sin(d_phi).unsqueeze(-1),
        torch.cos(d_phi).unsqueeze(-1),
        d_omega.unsqueeze(-1),
        omega_prod.unsqueeze(-1),
        beta_i_j.unsqueeze(-1),
        beta_j_i.unsqueeze(-1),
        cos_psi_c.unsqueeze(-1),
        axis_to_c.unsqueeze(2).expand(b, n, n).unsqueeze(-1),
        axis_to_c.unsqueeze(1).expand(b, n, n).unsqueeze(-1),
        h_i,
        h_j,
    ]
    packet = torch.cat(columns, dim=-1)
    assert packet.shape == (b, n, n, packet_width(d)), (
        f"packet must have shape (B, N, N, {packet_width(d)}), got {tuple(packet.shape)}"
    )
    return packet


def self_packet(c: Constellation, h_c_proj: Tensor) -> Tensor:
    # step 1: require a valid constellation and the caller's projection of h_c.
    assert_valid(c)
    b, n, d = int(c.x.shape[0]), int(c.x.shape[1]), int(c.h.shape[-1])
    assert h_c_proj.shape == (b, d), (
        f"h_c_proj must have shape (B, {d}), got {tuple(h_c_proj.shape)}"
    )

    # step 2: satellite-to-controller geometry under the section 3 convention:
    # phi_c is 0, omega_c is 0, u_c points from x_c to x_i, s_c is 1.
    rel = c.x - c.x_c.unsqueeze(1)  # from x_c to x_i
    dist = rel.norm(dim=-1).clamp(min=EPS)
    u_c = rel / dist.unsqueeze(-1)
    rho, sigma = size_pair(c.s, torch.ones_like(c.s))
    alpha = axis_alignment(c.u, u_c)
    # r_hat for the (i, C) pair is unit(x_i - x_c), which is u_c itself, so
    # beta copies alpha by the SPEC definition beta_{i<-j} = <u_i, r_hat_ij>.
    beta = alpha
    # the controller axis points along the ray from x_c to x_i, so its own
    # alignment with the sight line is exactly 1 by the convention.
    zeros = torch.zeros_like(c.s)
    ones = torch.ones_like(c.s)
    h_c_pair = h_c_proj.unsqueeze(1).expand(b, n, d)

    # step 3: stack in PacketColumn order and cat once.
    columns = [
        dist.unsqueeze(-1),
        torch.log(dist + EPS).unsqueeze(-1),
        rho.unsqueeze(-1),
        sigma.unsqueeze(-1),
        alpha.unsqueeze(-1),
        torch.sin(c.phi).unsqueeze(-1),
        torch.cos(c.phi).unsqueeze(-1),
        (c.omega / OMEGA_MAX).unsqueeze(-1),
        zeros.unsqueeze(-1),
        beta.unsqueeze(-1),
        beta.unsqueeze(-1),
        ones.unsqueeze(-1),
        beta.unsqueeze(-1),
        ones.unsqueeze(-1),
        c.h,
        h_c_pair,
    ]
    packet = torch.cat(columns, dim=-1)
    assert packet.shape == (b, n, packet_width(d)), (
        f"self packet must have shape (B, N, {packet_width(d)}), got {tuple(packet.shape)}"
    )
    return packet

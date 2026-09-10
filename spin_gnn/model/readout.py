# the readout: heads on the controller state and the SPEC predicate set.
# spec: docs/SPEC.md. every predicate is an invariant of the rigid state, so
# rotations cannot flip them; they live under no_grad and never enter a loss.
# flow:
# 1. predicates computes the six boolean fields and three integer counts.
# 2. ContinuousHead and DiscreteHead read the invariant h_c, so y and z are too.

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from spin_gnn.constants import TAU_D, TAU_OMEGA, TAU_PHI, TAU_U, TAU_W
from spin_gnn.constellation import assert_valid
from spin_gnn.geometry.box import wall_features
from spin_gnn.geometry.spin import phase_difference
from spin_gnn.layers.message import make_mlp
from spin_gnn.types import Constellation


@dataclass(frozen=True)
class Predicates:
    # the frozen product of one predicates call; the caller narrows on the fields.
    near: Tensor  # (B, N) bool: within TAU_D of the controller
    fast: Tensor  # (B, N) bool: |omega| above TAU_OMEGA
    corner: Tensor  # (B, N) bool: some wall feature below TAU_W
    aligned: Tensor  # (B, N, N) bool: |<u_i, u_j>| above TAU_U, diagonal False
    phaselock: Tensor  # (B, N, N) bool: aligned and |dphi| below TAU_PHI, diagonal False
    opp_spin: Tensor  # (B, N, N) bool: aligned with opposite spin signs, diagonal False
    n_near: Tensor  # (B,) int64: count of near satellites
    n_fast_near: Tensor  # (B,) int64: count of satellites both fast and near
    n_aligned_pairs: Tensor  # (B,) int64: count of unordered aligned pairs


def _pair_fields(c: Constellation) -> tuple[Tensor, Tensor, Tensor]:
    # step 1: broadcast the axis, speed, and phase fields into ordered pairs.
    b, n = int(c.x.shape[0]), int(c.x.shape[1])
    u_i = c.u.unsqueeze(2).expand(b, n, n, 3)
    u_j = c.u.unsqueeze(1).expand(b, n, n, 3)
    phi_i = c.phi.unsqueeze(2).expand(b, n, n)
    phi_j = c.phi.unsqueeze(1).expand(b, n, n)
    omega_i = c.omega.unsqueeze(2).expand(b, n, n)
    omega_j = c.omega.unsqueeze(1).expand(b, n, n)
    cos = (u_i * u_j).sum(dim=-1)
    return cos, phase_difference(phi_i, phi_j), omega_i * omega_j


@torch.no_grad()
def predicates(c: Constellation) -> Predicates:
    # step 1: require a valid constellation; predicates never require gradient.
    assert_valid(c)
    b, n = int(c.x.shape[0]), int(c.x.shape[1])
    eye_n = torch.eye(n, dtype=torch.bool, device=c.x.device)

    # step 2: the three per-satellite predicates per the SPEC thresholds.
    rel_c = c.x - c.x_c.unsqueeze(1)
    near = rel_c.norm(dim=-1) < TAU_D
    fast = c.omega.abs() > TAU_OMEGA
    corner = wall_features(c.x).amin(dim=-1) < TAU_W
    assert near.shape == (b, n) and near.dtype == torch.bool
    assert fast.shape == (b, n) and fast.dtype == torch.bool
    assert corner.shape == (b, n) and corner.dtype == torch.bool

    # step 3: the three pair predicates; each diagonal forced False so no
    # satellite pairs with itself.
    cos, d_phi, omega_prod = _pair_fields(c)
    aligned = (cos.abs() > TAU_U) & ~eye_n.unsqueeze(0)
    phaselock = aligned & (d_phi.abs() < TAU_PHI)
    opp_spin = aligned & (omega_prod < 0)
    assert aligned.shape == (b, n, n)
    assert not bool(aligned.diagonal(dim1=1, dim2=2).any())
    assert not bool(phaselock.diagonal(dim1=1, dim2=2).any())
    assert not bool(opp_spin.diagonal(dim1=1, dim2=2).any())

    # step 4: the counts; each aligned pair is counted once per direction.
    n_near = near.sum(dim=-1)
    n_fast_near = (fast & near).sum(dim=-1)
    n_aligned_pairs = aligned.sum(dim=(-1, -2)) // 2
    assert n_near.dtype == torch.int64
    assert bool((n_aligned_pairs * 2 == aligned.sum(dim=(-1, -2))).all())

    return Predicates(
        near=near,
        fast=fast,
        corner=corner,
        aligned=aligned,
        phaselock=phaselock,
        opp_spin=opp_spin,
        n_near=n_near,
        n_fast_near=n_fast_near,
        n_aligned_pairs=n_aligned_pairs,
    )


class ContinuousHead(nn.Module):
    # the regression head: h_c is invariant, so the readout is too.
    def __init__(self, d_c: int, k_out: int) -> None:
        super().__init__()
        assert d_c >= 1 and k_out >= 1, "head widths must be positive"
        self.k_out: int = k_out
        # step 1: a layer norm in front keeps the head input O(1) after the
        # residual stack; h_c is invariant, so the normed h_c is too.
        self.net: nn.Sequential = nn.Sequential(nn.LayerNorm(d_c), *make_mlp(d_c, d_c, k_out))

    def forward(self, h_c: Tensor) -> Tensor:
        y = self.net(h_c)
        assert y.shape == (h_c.shape[0], self.k_out)
        return y


class DiscreteHead(nn.Module):
    # the classification head: logits are invariant because h_c is invariant.
    def __init__(self, d_c: int, k_classes: int) -> None:
        super().__init__()
        assert d_c >= 1 and k_classes >= 1, "head widths must be positive"
        self.k_classes: int = k_classes
        self.net: nn.Sequential = nn.Sequential(
            nn.LayerNorm(d_c), *make_mlp(d_c, d_c, k_classes)
        )

    def forward(self, h_c: Tensor) -> Tensor:
        z = self.net(h_c)
        assert z.shape == (h_c.shape[0], self.k_classes)
        return z

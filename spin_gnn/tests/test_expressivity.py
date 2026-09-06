"""Prop 8, lower bound witness."""

# flow:
# 1. the multiset example pins the homometric equality in float64.
# 2. the model separates the pair through the angular packet columns.
# 3. a distance-only stand-in, blind to those columns, fails to separate.

import torch

from spin_gnn.constants import EPS
from spin_gnn.data.degenerate_pairs import distance_multiset, homometric_pair
from spin_gnn.layers.message import make_mlp
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.model.spin_gnn_gnn import build_model


def _witness_config() -> SpinGnnConfig:
    # n = 6 is forced by the witness; the other widths stay small and legal.
    return SpinGnnConfig(n=6, d=16, c=4, d_c=32, d_m=16, n_layers=2)


def test_distance_multisets_match() -> None:
    config = _witness_config()
    a, b = homometric_pair(config)
    gap = (distance_multiset(a) - distance_multiset(b)).abs().max()
    assert bool(gap <= 1e-12), f"multisets differ by {float(gap)}"


def test_model_separates_the_homometric_pair() -> None:
    config = _witness_config()
    a, b = homometric_pair(config)
    model = build_model(config).double()
    h_c_a = model(a).constellation.h_c
    h_c_b = model(b).constellation.h_c
    gap = (h_c_a - h_c_b).norm().detach()
    print(f"\nmodel separation of the homometric pair: {float(gap):.6e}")
    assert float(gap) > 1e-6


def test_distance_only_stand_in_cannot_separate() -> None:
    # a stand-in that sums make_mlp over [d_ij, log d_ij + EPS] sees only the
    # distance multiset, so it must land the same value on both clouds.
    config = _witness_config()
    a, b = homometric_pair(config)
    torch.manual_seed(0)
    stand_in = make_mlp(2, 16, 8).double()

    def score(x: torch.Tensor) -> torch.Tensor:
        n = int(x.shape[1])
        rel = x.unsqueeze(2) - x.unsqueeze(1)
        dist = rel.norm(dim=-1)
        upper = dist[:, torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)]
        feats = torch.stack((upper, torch.log(upper + EPS)), dim=-1)
        return stand_in(feats).sum(dim=1)

    gap = (score(a.x) - score(b.x)).abs().max()
    assert bool(gap <= 1e-9), f"distance-only stand-in separated by {float(gap)}"

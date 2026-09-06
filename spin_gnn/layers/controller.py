# the controller pool: mean, max, and attention over satellites. spec: docs/SPEC.md.
# flow:
# 1. attention scores come from an invariant five-scalar row per satellite.
# 2. pool is mean(h), max(h), attention(h), mean(s), mean(|omega|)/OMEGA_MAX, mean(d_iC).
# 3. every input is invariant, so the pool is invariant.

import torch
from torch import Tensor, nn

from spin_gnn.constants import OMEGA_MAX
from spin_gnn.constellation import assert_valid
from spin_gnn.layers.message import make_mlp
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.types import Constellation


def _attention_row(c: Constellation) -> Tensor:
    # step 1: five invariant scalars per satellite, all from the legal packet stock.
    rel_c = c.x - c.x_c.unsqueeze(1)
    dist_c = rel_c.norm(dim=-1)
    row = torch.stack(
        (
            c.s,
            c.omega / OMEGA_MAX,
            torch.sin(c.phi),
            torch.cos(c.phi),
            dist_c,
        ),
        dim=-1,
    )
    assert row.shape[-1] == 5
    return row


class ControllerPool(nn.Module):
    def __init__(self, config: SpinGnnConfig) -> None:
        super().__init__()
        self.config: SpinGnnConfig = config
        # step 1: the logit scorer for attention over satellites.
        self.mlp_l: nn.Sequential = make_mlp(5, config.d_m, 1)

    def forward(self, c: Constellation) -> Tensor:
        # step 1: require a valid constellation with the configured scalar width.
        assert_valid(c)
        b, d = int(c.x.shape[0]), self.config.d
        assert c.h.shape[-1] == d, f"h width must be {d}, got {c.h.shape[-1]}"

        # step 2: the three h pools. attention weighs satellites by the scorer.
        logits = self.mlp_l(_attention_row(c)).squeeze(-1)  # (B, N)
        weights = torch.softmax(logits, dim=-1)
        attn = (weights.unsqueeze(-1) * c.h).sum(dim=1)
        mean_h = c.h.mean(dim=1)
        max_h = c.h.amax(dim=1)

        # step 3: the three summary scalars.
        rel_c = c.x - c.x_c.unsqueeze(1)
        dist_c = rel_c.norm(dim=-1)
        scalars = torch.stack(
            (c.s.mean(dim=1), c.omega.abs().mean(dim=1) / OMEGA_MAX, dist_c.mean(dim=1)),
            dim=-1,
        )

        pool = torch.cat((mean_h, max_h, attn, scalars), dim=-1)
        assert pool.shape == (b, 3 * d + 3)
        return pool

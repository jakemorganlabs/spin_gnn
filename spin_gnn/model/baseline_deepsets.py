# the controller-free DeepSets baseline over the invariant scalar packet.
# reference: Zaheer et al. 2017, Deep Sets. spec: docs/SPEC.md.
# flow:
# 1. BaselineDeepSets embeds each off-diagonal edge with phi_edge and means.
# 2. match_width searches the smallest width inside PARAM_MATCH_TOL of a target.
# 3. build_baseline seeds torch and returns the width-matched baseline.

import torch
from torch import Tensor, nn

from spin_gnn.constants import PARAM_MATCH_TOL, SEED
from spin_gnn.constellation import assert_valid
from spin_gnn.geometry.invariants import N_SCALAR_COLUMNS, edge_packet
from spin_gnn.layers.message import make_mlp
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.types import Constellation

# search bounds: the width that hits the parameter target always sits inside.
_WIDTH_MIN: int = 8
_WIDTH_MAX: int = 2048


class BaselineDeepSets(nn.Module):
    # deep sets over the N(N-1) off-diagonal edges; no controller node.
    def __init__(self, width: int, k_classes: int = 4) -> None:
        super().__init__()
        assert width >= 1 and k_classes >= 1, "width and k_classes must be positive"
        self.width: int = width
        self.k_classes: int = k_classes
        self.phi_edge: nn.Sequential = make_mlp(N_SCALAR_COLUMNS, width, width)
        self.rho: nn.Sequential = make_mlp(width, width, k_classes)

    def forward(self, raw: Constellation) -> Tensor:
        # step 1: require a valid constellation; the packet reads the geometry.
        assert_valid(raw)
        b, n = int(raw.x.shape[0]), int(raw.x.shape[1])

        # step 2: slice the 14 scalar columns, embed each edge, mask the diagonal.
        packet = edge_packet(raw)[..., :N_SCALAR_COLUMNS]
        embedded = self.phi_edge(packet)
        off_diag = ~torch.eye(n, dtype=torch.bool, device=raw.x.device)
        mask = off_diag.unsqueeze(0).unsqueeze(-1).to(embedded.dtype)
        summed = (embedded * mask).sum(dim=(1, 2))
        pooled = summed / mask.sum(dim=(1, 2)).clamp(min=1.0)

        # step 3: rho maps the pooled edge summary to the class logits.
        out = self.rho(pooled)
        assert out.shape == (b, self.k_classes)
        return out


def _count_baseline(width: int, k_classes: int) -> int:
    # step 1: build only to count; no seed matters because shapes fix the count.
    return sum(p.numel() for p in BaselineDeepSets(width, k_classes).parameters())


def match_width(target_params: int, k_classes: int = 4) -> int:
    # step 1: a positive target is required before any width can match it.
    assert target_params > 0, f"target_params must be positive, got {target_params}"

    # step 2: the count is increasing in width, so a single pass finds the
    # nearest width; ties keep the smaller width already recorded.
    best_width = -1
    best_gap = float("inf")
    for width in range(_WIDTH_MIN, _WIDTH_MAX + 1):
        gap = abs(_count_baseline(width, k_classes) - target_params)
        if gap < best_gap:
            best_gap = gap
            best_width = width
    assert best_gap <= PARAM_MATCH_TOL * target_params, (
        f"no width in [{_WIDTH_MIN}, {_WIDTH_MAX}] lands within "
        f"{PARAM_MATCH_TOL} of {target_params}"
    )
    return best_width


def build_baseline(
    config: SpinGnnConfig, target_params: int, k_classes: int = 4
) -> BaselineDeepSets:
    # step 1: the config seeds the draw; the target fixes the width.
    torch.manual_seed(config.seed if config.seed is not None else SEED)
    width = match_width(target_params, k_classes)
    return BaselineDeepSets(width, k_classes)

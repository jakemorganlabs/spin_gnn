# the diagnostics: the measured per-layer step bound and the model symmetry gap.
# spec: docs/SPEC.md. Prop 3 cites measure_max_step; Prop 4 cites symmetry_gap.
# flow:
# 1. measure_max_step re-runs the layers per satellite step with debug on and
#    takes the largest position move over batches, layers, and satellites.
# 2. symmetry_gap compares h_c between the rotated and the unrotated inputs.

from dataclasses import dataclass

import torch
from torch import Tensor

from spin_gnn.constants import MARGIN
from spin_gnn.constellation import assert_valid, in_interior, rotate_about_controller
from spin_gnn.model.spin_gnn_gnn import SpinGnn
from spin_gnn.types import Constellation


@dataclass(frozen=True)
class MaxStepReport:
    # the frozen product of one measure_max_step call.
    max_step: float
    per_layer: tuple[float, ...]
    bound: float  # MARGIN / n_layers
    within_bound: bool


@torch.no_grad()
def measure_max_step(model: SpinGnn, batches: list[Constellation]) -> MaxStepReport:
    # step 1: every batch must be valid and fully interior, so the only bound
    # that matters is the one Prop 3 states.
    assert len(batches) >= 1, "measure_max_step needs at least one batch"
    for c in batches:
        assert_valid(c)
        assert bool(in_interior(c).all()), "every batch must be interior"
    n_layers = model.config.n_layers
    per_layer_acc: list[list[float]] = [[] for _ in range(n_layers)]

    # step 2: walk the layers, recording the largest single-layer move.
    for c in batches:
        states = model.forward_layers(c)
        for index in range(n_layers):
            before = states[index]
            after = states[index + 1]
            step = (after.x - before.x).norm(dim=-1).max()
            per_layer_acc[index].append(float(step))

    per_layer = tuple(max(bucket) for bucket in per_layer_acc)
    max_step = max(per_layer)
    bound = MARGIN / n_layers
    return MaxStepReport(
        max_step=max_step,
        per_layer=per_layer,
        bound=bound,
        within_bound=max_step < bound,
    )


@torch.no_grad()
def symmetry_gap(model: SpinGnn, c: Constellation, rotation: Tensor) -> Tensor:
    # step 1: one rotation per batch element; compare the invariant h_c.
    assert_valid(c)
    assert rotation.shape == (c.x.shape[0], 3, 3), (
        f"rotation must have shape (B, 3, 3), got {tuple(rotation.shape)}"
    )
    dets = torch.linalg.det(rotation.to(torch.float64))
    assert bool(((dets - 1).abs() <= 1e-6).all()), "each rotation must have det 1"
    before = model(c).constellation.h_c
    after = model(rotate_about_controller(c, rotation)).constellation.h_c
    gap = (before - after).abs().amax(dim=-1)
    assert gap.shape == (c.x.shape[0],)
    assert bool((gap >= 0).all())
    return gap

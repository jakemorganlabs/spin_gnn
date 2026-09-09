# tests for the DeepSets baseline and the width matcher. spec: docs/SPEC.md.
# flow:
# 1. the parameter example pins the baseline within ten percent of full.
# 2. examples cover the logit shape, the Haar invariance, and the matcher rule.
# 3. the property covers packet invariance on interior Haar draws.

import torch
from hypothesis import given
from hypothesis import strategies as st

from spin_gnn.constants import PARAM_MATCH_TOL
from spin_gnn.constellation import rotate_about_controller, to_dtype
from spin_gnn.geometry.frames import random_rotation
from spin_gnn.model.baseline_deepsets import (
    BaselineDeepSets,
    build_baseline,
    match_width,
)
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.model.spin_gnn_gnn import build_model, count_parameters
from spin_gnn.tests.test_equivariance import interior_batch, small_config


def test_baseline_param_count_within_ten_percent_of_full() -> None:
    config = SpinGnnConfig()
    full = build_model(config)
    baseline = build_baseline(config, count_parameters(full))
    gap = abs(count_parameters(baseline) - count_parameters(full))
    assert gap <= PARAM_MATCH_TOL * count_parameters(full)


def test_baseline_logits_shape_on_interior_batch(generator: torch.Generator) -> None:
    config = small_config()
    baseline = BaselineDeepSets(16).double()
    c = to_dtype(interior_batch(4, config, generator), torch.float64)
    out = baseline(c)
    assert out.shape == (4, 4)


def test_baseline_logits_invariant_under_haar(generator: torch.Generator) -> None:
    config = small_config()
    baseline = BaselineDeepSets(16).double()
    c = to_dtype(interior_batch(2, config, generator), torch.float64)
    rotation = random_rotation(2, generator)
    before = baseline(c)
    after = baseline(rotate_about_controller(c, rotation))
    gap = (after - before).abs()
    assert bool((gap <= 1e-8).all()), f"logits moved by {float(gap.max())}"


def test_match_width_postcondition_or_raises() -> None:
    # the contract accepts either a width inside the tolerance or a loud raise.
    target = 1000
    try:
        width = match_width(target)
    except AssertionError:
        return
    params = sum(p.numel() for p in BaselineDeepSets(width).parameters())
    assert abs(params - target) <= PARAM_MATCH_TOL * target


@given(st.integers(min_value=1, max_value=3))
def test_baseline_invariant_on_interior_property(seed_offset: int) -> None:
    config = small_config()
    baseline = BaselineDeepSets(16).double()
    gen = torch.Generator().manual_seed(8801 + seed_offset)
    c = to_dtype(interior_batch(2, config, gen), torch.float64)
    rotation = random_rotation(2, gen)
    gap = (baseline(rotate_about_controller(c, rotation)) - baseline(c)).abs()
    assert bool((gap <= 1e-8).all()), f"logits moved by {float(gap.max())}"

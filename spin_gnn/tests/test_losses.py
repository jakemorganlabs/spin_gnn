# tests for the task losses and the three regularizers. spec: docs/SPEC.md.
# flow:
# 1. examples cover zero regularizers on a valid sample and positive loss when
#    one constraint breaks.
# 2. examples cover the classification and regression tasks.
# 3. the example and the property pin the weighted total to its terms.
# 4. the property covers box and spin both zero on every valid draw.

import torch
from hypothesis import given
from hypothesis import strategies as st

from spin_gnn.constants import SEED, W_BOX, W_SPIN, W_UNIT
from spin_gnn.constellation import validate
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.train.losses import (
    loss_box,
    loss_classification,
    loss_regression,
    loss_spin,
    loss_unit,
    total_loss,
)
from spin_gnn.train.tasks_synthetic import sample_interior
from spin_gnn.types import Constellation, ValidOutcome


def _edited(c: Constellation, **fields: torch.Tensor) -> Constellation:
    # frozen dataclasses reject assignment; rebuild with the field swapped.
    data = dict(
        x=c.x, s=c.s, u=c.u, phi=c.phi, omega=c.omega,
        h=c.h, v=c.v, x_c=c.x_c, h_c=c.h_c,
    )
    data.update(fields)
    return Constellation(**data)  # type: ignore[arg-type]


def test_valid_interior_sample_has_zero_box_and_spin_and_small_unit(
    generator: torch.Generator,
) -> None:
    c = sample_interior(4, SpinGnnConfig(), generator, dtype=torch.float64)
    assert isinstance(validate(c), ValidOutcome)
    assert float(loss_box(c)) == 0.0
    assert float(loss_spin(c)) == 0.0
    assert float(loss_unit(c)) < 1e-6


def test_scaled_axis_raises_unit(generator: torch.Generator) -> None:
    c = sample_interior(4, SpinGnnConfig(), generator, dtype=torch.float64)
    u = c.u.clone()
    u[0, 0] = u[0, 0] * 2.0  # bypass validate: one axis norm is now 2
    out = loss_unit(_edited(c, u=u))
    assert float(out) > 0.0


def test_position_outside_box_raises_box(generator: torch.Generator) -> None:
    c = sample_interior(4, SpinGnnConfig(), generator, dtype=torch.float64)
    x = c.x.clone()
    x[0, 0, 0] = 1.2  # bypass validate: one coordinate sits past the far wall
    out = loss_box(_edited(c, x=x))
    assert float(out) > 0.0


def test_correct_logits_with_margin_20_are_below_1e_3() -> None:
    # a one-hot-correct logit vector with a margin of 20 scores near zero.
    z = torch.zeros(4, 4, dtype=torch.float64)
    y_true = torch.arange(4)
    z[torch.arange(4), y_true] = 20.0
    out = loss_classification(z, y_true)
    assert float(out) < 1e-3


def test_regression_of_exact_targets_is_zero() -> None:
    y = torch.randn(8, 3, dtype=torch.float64)
    out = loss_regression(y, y.clone())
    assert float(out) == 0.0


def test_total_loss_of_task_one_on_valid_sample_is_one(
    generator: torch.Generator,
) -> None:
    c = sample_interior(4, SpinGnnConfig(), generator, dtype=torch.float64)
    task = torch.tensor(1.0, dtype=torch.float64)
    breakdown = total_loss(task, c)
    assert abs(float(breakdown.total) - 1.0) <= 1e-6


def test_total_loss_breakdown_recovers_the_weighted_sum(
    generator: torch.Generator,
) -> None:
    c = sample_interior(4, SpinGnnConfig(), generator, dtype=torch.float64)
    task = torch.tensor(1.7, dtype=torch.float64)
    breakdown = total_loss(task, c)
    expected = (
        task
        + W_UNIT * breakdown.unit
        + W_BOX * breakdown.box
        + W_SPIN * breakdown.spin
    )
    assert abs(float(breakdown.total - expected)) <= 1e-9


@given(st.integers(min_value=1, max_value=6))
def test_valid_constellation_has_zero_box_and_spin(seed_offset: int) -> None:
    # Property 4. every valid sample already respects the box and the spin
    # ceiling, so both regularizers vanish exactly.
    generator = torch.Generator().manual_seed(SEED + 307 + seed_offset)
    c = sample_interior(4, SpinGnnConfig(), generator, dtype=torch.float64)
    assert isinstance(validate(c), ValidOutcome)
    assert float(loss_box(c)) == 0.0
    assert float(loss_spin(c)) == 0.0

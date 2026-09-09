# tests for the training loop, the bootstrap, and the ablation. spec: docs/SPEC.md.
# flow:
# 1. examples cover evaluate, the smoke training runs, and the record round trip.
# 2. examples cover the bootstrap constant and bounded cases, and summarize.
# 3. the ablation smoke run writes every listed file with four entries.
# 4. the property pins bootstrap_ci inside [min, max] with low at most high.

import json

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st

from spin_gnn.constants import SEED, SMOKE_STEPS
from spin_gnn.tests.test_equivariance import small_config
from spin_gnn.train.ablate import bootstrap_ci, run_ablation, summarize
from spin_gnn.train.loop import RunResult, evaluate, train_task_b
from spin_gnn.train.tasks_synthetic import construct_task_b_flat


class _FixedLogits(torch.nn.Module):
    # a module that returns the same logit matrix regardless of the input.
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits: torch.Tensor = logits

    def forward(self, c: object) -> torch.Tensor:
        return self.logits


def test_evaluate_returns_one_when_logits_match_labels(generator: torch.Generator) -> None:
    # a one-hot-correct logit map scores a perfect held-out fraction.
    c, labels = construct_task_b_flat(4, small_config(), generator)
    z = torch.zeros(4, 4)
    z[torch.arange(4), labels] = 20.0
    out = evaluate(_FixedLogits(z), c, labels, is_baseline=True)
    assert out == 1.0


def test_smoke_train_full_run_result_is_valid(generator: torch.Generator) -> None:
    result = train_task_b("full", small_config(), seed=0, steps=SMOKE_STEPS)
    assert result.steps == SMOKE_STEPS
    assert result.steps_to_milestone is None or result.steps_to_milestone >= 1
    assert result.final_loss == result.final_loss
    assert isinstance(result.within_bound_after, bool)
    assert 0.0 <= result.heldout_acc <= 1.0


def test_smoke_train_baseline_has_no_step_bound(generator: torch.Generator) -> None:
    result = train_task_b("baseline", small_config(), seed=0, steps=SMOKE_STEPS)
    assert result.max_step_after is None
    assert result.within_bound_after is None
    assert result.final_loss == result.final_loss


def test_run_result_round_trip(generator: torch.Generator) -> None:
    result = train_task_b("full", small_config(), seed=0, steps=SMOKE_STEPS)
    assert RunResult.model_validate_json(result.model_dump_json()) == result


def test_bootstrap_constant_list_is_that_constant() -> None:
    rng = np.random.default_rng(SEED)
    assert bootstrap_ci([0.9, 0.9, 0.9], rng) == (0.9, 0.9)


def test_bootstrap_interval_sits_inside_min_max() -> None:
    rng = np.random.default_rng(SEED)
    low, high = bootstrap_ci([0.8, 0.9, 1.0], rng)
    assert 0.8 <= low <= high <= 1.0


def _result(run_name: str, seed: int, milestone: int | None) -> RunResult:
    return RunResult(
        run_name=run_name,  # type: ignore[arg-type]
        seed=seed,
        steps=SMOKE_STEPS,
        train_acc=0.5,
        heldout_acc=0.5,
        steps_to_milestone=milestone,
        params=100,
        sec_per_step=0.01,
        max_step_after=None if run_name == "baseline" else 0.01,
        within_bound_after=None if run_name == "baseline" else True,
        final_loss=1.0,
    )


def test_summarize_counts_missing_and_pairs() -> None:
    # four runs over two seeds; the scalar_only seed 1 milestone is missing, so
    # one run is missing and only one seed pairs full against scalar_only.
    results = [
        _result("full", 0, 10),
        _result("full", 1, 12),
        _result("static", 0, 11),
        _result("static", 1, 13),
        _result("scalar_only", 0, 20),
        _result("scalar_only", 1, None),
        _result("baseline", 0, 30),
        _result("baseline", 1, 31),
    ]
    summaries, paired = summarize(results)
    by_name = {s.run_name: s for s in summaries}
    assert by_name["scalar_only"].milestone_missing == 1
    assert paired.n_pairs == 1
    assert paired.diff_mean is not None


def test_ablate_smoke_writes_every_file(tmp_path) -> None:
    run_ablation([0], SMOKE_STEPS, tmp_path, small_config())
    for name in ("task_b.json", "summary.json", "summary.md"):
        assert (tmp_path / name).exists()
    for fig in (
        "class_0.png", "class_1.png", "class_2.png", "class_3.png",
        "layers.png", "symmetry_gap.png",
    ):
        assert (tmp_path / "figures" / fig).exists()
    entries = json.loads((tmp_path / "task_b.json").read_text())
    assert len(entries) == 4


@given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=10))
def test_bootstrap_ci_bounded_and_ordered(values: list[float]) -> None:
    rng = np.random.default_rng(SEED)
    low, high = bootstrap_ci(values, rng)
    assert low <= high
    assert min(values) - 1e-9 <= low
    assert high <= max(values) + 1e-9

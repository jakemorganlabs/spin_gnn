# the Task B training loop and the per-run record. spec: docs/SPEC.md.
# flow:
# 1. build the model for the run name, seeding torch per the config.
# 2. draw one fixed held-out set with seed offset off the training stream.
# 3. loop: draw a flat batch, forward, total_loss, backward, step, and every
#    EVAL_EVERY record held-out accuracy and the first milestone step.
# 4. re-measure the step bound on interior batches and record the run.

import json
import time
from typing import Literal

import torch
from pydantic import BaseModel
from torch import nn

from spin_gnn.constants import (
    ACC_MILESTONE,
    BATCH_SIZE,
    EVAL_EVERY,
    HELD_OUT_SEED_OFFSET,
    HELD_OUT_SIZE,
    LR,
    MAX_STEP_BATCHES,
    WEIGHT_DECAY,
)
from spin_gnn.constellation import assert_valid
from spin_gnn.model.baseline_deepsets import build_baseline
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.model.diagnostics import measure_max_step
from spin_gnn.model.spin_gnn_gnn import SpinGnn, build_model, count_parameters
from spin_gnn.train.losses import loss_classification, total_loss
from spin_gnn.train.tasks_synthetic import construct_task_b_flat, sample_interior
from spin_gnn.types import Constellation

RunName = Literal["full", "static", "scalar_only", "baseline"]

_RUN_NAMES: tuple[str, ...] = ("full", "static", "scalar_only", "baseline")


class RunResult(BaseModel, frozen=True):
    # the per-run record; frozen so the ablation serializes it verbatim.
    run_name: RunName
    seed: int
    steps: int
    train_acc: float
    heldout_acc: float
    steps_to_milestone: int | None
    params: int
    sec_per_step: float
    max_step_after: float | None  # None for the baseline
    within_bound_after: bool | None
    final_loss: float


def _log(stage: str, status: str, **fields: float | int | str | None) -> None:
    # step 1: one structured line per stage event, json on stdout.
    record = {"stage": stage, "status": status, **fields}
    print(json.dumps(record))


def _build_run(run_name: str, config: SpinGnnConfig) -> nn.Module:
    # step 1: the full, static, and scalar-only runs share the model class;
    # the baseline is width-matched against the full parameter count.
    if run_name == "full":
        return build_model(config)
    if run_name == "static":
        return build_model(config.static())
    if run_name == "scalar_only":
        return build_model(config.scalar_only())
    target = count_parameters(build_model(config))
    return build_baseline(config, target)


def _logits(
    model: nn.Module, c: Constellation, is_baseline: bool
) -> torch.Tensor:
    # step 1: the baseline returns logits directly; the model returns them on z.
    if is_baseline:
        logits = model(c)
        assert logits.ndim == 2, f"baseline logits must be 2-d, got {tuple(logits.shape)}"
        return logits
    out = model(c)
    assert hasattr(out, "z"), "the Spin_GNN forward must return a ModelOutput with z"
    return out.z


def evaluate(
    model: nn.Module, c: Constellation, labels: torch.Tensor, is_baseline: bool
) -> float:
    # step 1: the fraction of correct argmax labels in [0, 1], under eval mode.
    assert_valid(c)
    model.eval()
    with torch.no_grad():
        logits = _logits(model, c, is_baseline)
        correct = (logits.argmax(dim=-1) == labels).to(torch.float64).mean()
    out = float(correct)
    assert 0.0 <= out <= 1.0
    return out


def train_task_b(run_name: str, config: SpinGnnConfig, seed: int, steps: int) -> RunResult:
    # step 1: pin the contract before any build work.
    assert steps >= 1, f"steps must be at least 1, got {steps}"
    assert run_name in _RUN_NAMES, f"run_name must be one of {_RUN_NAMES}, got {run_name}"
    is_baseline = run_name == "baseline"
    _log("train_task_b", "start", run=run_name, seed=seed, steps=steps)

    # step 2: one data generator per run; the held-out set sits off that stream.
    data_gen = torch.Generator().manual_seed(seed)
    held_out_gen = torch.Generator().manual_seed(seed + HELD_OUT_SEED_OFFSET)
    held_out_c, held_out_labels = construct_task_b_flat(
        HELD_OUT_SIZE, config, held_out_gen
    )
    model = _build_run(run_name, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # step 3: the training loop; time it with perf_counter over the steps.
    model.train()
    steps_to_milestone: int | None = None
    train_acc = 0.0
    heldout_acc = evaluate(model, held_out_c, held_out_labels, is_baseline)
    final_loss = float("nan")
    start = time.perf_counter()
    for step in range(1, steps + 1):
        batch, labels = construct_task_b_flat(BATCH_SIZE, config, data_gen)
        logits = _logits(model, batch, is_baseline)
        task = loss_classification(logits, labels)
        if is_baseline:
            # step 3a: the baseline sees the task loss only; it has no updates
            # to regularize.
            loss = task
        else:
            loss = total_loss(task, model(batch).constellation).total
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        train_acc = float((logits.argmax(dim=-1) == labels).to(torch.float64).mean())

        if step % EVAL_EVERY == 0:
            heldout_acc = evaluate(model, held_out_c, held_out_labels, is_baseline)
            if steps_to_milestone is None and heldout_acc >= ACC_MILESTONE:
                steps_to_milestone = step
    # step 4: close the loop with a final held-out read and the elapsed time.
    heldout_acc = evaluate(model, held_out_c, held_out_labels, is_baseline)
    if steps_to_milestone is None and heldout_acc >= ACC_MILESTONE:
        steps_to_milestone = steps
    elapsed = time.perf_counter() - start
    sec_per_step = elapsed / steps

    # step 5: the step bound is a property of the model only, not the baseline.
    max_step_after: float | None = None
    within_bound_after: bool | None = None
    if not is_baseline:
        assert isinstance(model, SpinGnn)
        bound_gen = torch.Generator().manual_seed(seed + 2 * HELD_OUT_SEED_OFFSET)
        interior = [
            sample_interior(BATCH_SIZE, config, bound_gen)
            for _ in range(MAX_STEP_BATCHES)
        ]
        report = measure_max_step(model, interior)
        max_step_after = report.max_step
        within_bound_after = report.within_bound

    # step 6: the final loss must be a finite float before the record closes.
    assert final_loss == final_loss and abs(final_loss) != float("inf"), (
        f"final_loss must be finite, got {final_loss}"
    )
    result = RunResult(
        run_name=run_name,  # type: ignore[arg-type]
        seed=seed,
        steps=steps,
        train_acc=train_acc,
        heldout_acc=heldout_acc,
        steps_to_milestone=steps_to_milestone,
        params=count_parameters(model),
        sec_per_step=sec_per_step,
        max_step_after=max_step_after,
        within_bound_after=within_bound_after,
        final_loss=final_loss,
    )
    _log(
        "train_task_b",
        "success",
        run=run_name,
        seed=seed,
        heldout_acc=heldout_acc,
        steps_to_milestone=steps_to_milestone,
    )
    return result

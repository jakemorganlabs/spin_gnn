# the four-way seeded ablation with bootstrap confidence intervals and figures.
# spec: docs/SPEC.md. every number session 7 publishes is measured and written
# to results/ here; nothing in this module states a claim.
# flow:
# 1. Summary and PairedSummary record the per-config and paired statistics.
# 2. bootstrap_ci resamples a value list into a 95 percent interval.
# 3. summarize folds the run list into one Summary per config plus the
#    full-versus-scalar_only paired comparison.
# 4. run_ablation drives the four configs over the seeds, writes json after
#    every run, and writes the markdown table and the figures.

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel

from spin_gnn.constants import BOOTSTRAP_N, CI_HIGH, CI_LOW, SEED, SMOKE_STEPS
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.model.spin_gnn_gnn import build_model
from spin_gnn.train.loop import RunResult, train_task_b
from spin_gnn.train.tasks_synthetic import construct_task_b
from spin_gnn.viz.render_box import plot_constellation, plot_layers, plot_symmetry_gap

_RUN_ORDER: tuple[str, ...] = ("full", "static", "scalar_only", "baseline")


class Summary(BaseModel, frozen=True):
    # the per-configuration statistics over seeds.
    run_name: str
    n_seeds: int
    heldout_mean: float
    heldout_ci: tuple[float, float]
    milestone_mean: float | None
    milestone_ci: tuple[float, float] | None
    milestone_missing: int
    params: int
    sec_per_step_mean: float


class PairedSummary(BaseModel, frozen=True):
    # the paired full-versus-scalar_only difference in steps to milestone.
    a: str
    b: str
    diff_mean: float | None
    diff_ci: tuple[float, float] | None
    n_pairs: int


def bootstrap_ci(values: list[float], rng: np.random.Generator) -> tuple[float, float]:
    # step 1: resample the mean BOOTSTRAP_N times; read the two quantiles off.
    assert len(values) >= 1, "bootstrap_ci needs at least one value"
    arr = np.asarray(values, dtype=np.float64)
    low_value = float(arr.min())
    high_value = float(arr.max())
    if low_value == high_value:
        return low_value, high_value
    draws = rng.choice(arr, size=(BOOTSTRAP_N, arr.shape[0]), replace=True)
    means = draws.mean(axis=1)
    low = float(np.quantile(means, CI_LOW))
    high = float(np.quantile(means, CI_HIGH))
    assert low <= high
    return low, high


def summarize(results: list[RunResult]) -> tuple[list[Summary], PairedSummary]:
    # step 1: every run name sees the same seed set; check it before any mean.
    assert len(results) >= 1, "summarize needs at least one result"
    seed_set = sorted({r.seed for r in results})
    by_name: dict[str, list[RunResult]] = {name: [] for name in _RUN_ORDER}
    for r in results:
        by_name[r.run_name].append(r)
    for name in _RUN_ORDER:
        assert sorted(r.seed for r in by_name[name]) == seed_set, (
            f"run {name} must cover the same seed set"
        )

    rng = np.random.default_rng(SEED)
    summaries: list[Summary] = []
    for name in _RUN_ORDER:
        runs = by_name[name]
        heldout = [r.heldout_acc for r in runs]
        milestones = [r.steps_to_milestone for r in runs if r.steps_to_milestone is not None]
        milestone_mean = float(np.mean(milestones)) if milestones else None
        milestone_ci = bootstrap_ci([float(m) for m in milestones], rng) if milestones else None
        summaries.append(
            Summary(
                run_name=name,
                n_seeds=len(runs),
                heldout_mean=float(np.mean(heldout)),
                heldout_ci=bootstrap_ci(heldout, rng),
                milestone_mean=milestone_mean,
                milestone_ci=milestone_ci,
                milestone_missing=len(runs) - len(milestones),
                params=runs[0].params,
                sec_per_step_mean=float(np.mean([r.sec_per_step for r in runs])),
            )
        )

    # step 2: the paired difference pairs full with scalar_only on each seed
    # where both milestones exist; missing runs reduce the pair count.
    full_by_seed = {r.seed: r for r in by_name["full"]}
    scalar_by_seed = {r.seed: r for r in by_name["scalar_only"]}
    diffs: list[float] = []
    for seed in seed_set:
        f_m = full_by_seed[seed].steps_to_milestone
        s_m = scalar_by_seed[seed].steps_to_milestone
        if f_m is not None and s_m is not None:
            diffs.append(float(f_m - s_m))
    paired = PairedSummary(
        a="full",
        b="scalar_only",
        diff_mean=float(np.mean(diffs)) if diffs else None,
        diff_ci=bootstrap_ci(diffs, rng) if diffs else None,
        n_pairs=len(diffs),
    )
    return summaries, paired


def _summary_md(summaries: list[Summary], paired: PairedSummary) -> str:
    # step 1: one header row, then one honest row per configuration.
    header = (
        "| run | params | heldout mean | heldout 95% CI | steps to 0.90 mean "
        "| steps to 0.90 95% CI | missing | sec per step |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        milestone_mean = f"{s.milestone_mean:.1f}" if s.milestone_mean is not None else "null"
        if s.milestone_ci is not None:
            milestone_ci = f"[{s.milestone_ci[0]:.1f}, {s.milestone_ci[1]:.1f}]"
        else:
            milestone_ci = "null"
        lines.append(
            f"| {s.run_name} | {s.params} | {s.heldout_mean:.4f} | "
            f"[{s.heldout_ci[0]:.4f}, {s.heldout_ci[1]:.4f}] | "
            f"{milestone_mean} | {milestone_ci} | {s.milestone_missing} | "
            f"{s.sec_per_step_mean:.4f} |"
        )

    # step 2: one line for the paired difference; null when no seed pairs.
    if paired.diff_mean is not None and paired.diff_ci is not None:
        lines.append(
            f"\npaired steps-to-milestone difference ({paired.a} - {paired.b}) over "
            f"{paired.n_pairs} paired seeds: mean {paired.diff_mean:.1f}, "
            f"95% CI [{paired.diff_ci[0]:.1f}, {paired.diff_ci[1]:.1f}]."
        )
    else:
        lines.append(
            f"\npaired steps-to-milestone difference ({paired.a} - {paired.b}): "
            f"no paired seeds reached the milestone."
        )
    return "\n".join(lines) + "\n"


def _write_figures(out_dir: Path, config: SpinGnnConfig, seed: int) -> None:
    # step 1: one constellation per class, drawn with a dedicated generator.
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for class_id in range(4):
        gen = torch.Generator().manual_seed(seed + 1000 + class_id)
        c = construct_task_b(1, class_id, config, gen)
        plot_constellation(c, 0, figures / f"class_{class_id}.png")

    # step 2: the layer panels read the trained full model of seed 0; the
    # smoke path hands in the freshly built one because 20 steps do not move it.
    layer_model = build_model(config)
    layer_gen = torch.Generator().manual_seed(seed + 2000)
    layer_c = construct_task_b(1, 0, config, layer_gen)
    plot_layers(layer_model, layer_c, 0, figures / "layers.png")

    # step 3: the symmetry gap curve reads a fresh model, per the plan.
    gap_model = build_model(config)
    gap_gen = torch.Generator().manual_seed(seed + 3000)
    plot_symmetry_gap(gap_model, config, gap_gen, figures / "symmetry_gap.png")


def run_ablation(seeds: list[int], steps: int, out_dir: Path, config: SpinGnnConfig) -> None:
    # step 1: seeds in order, run names in order; write partial json per run so
    # a crash keeps the completed rows.
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    for seed in seeds:
        for run_name in _RUN_ORDER:
            results.append(train_task_b(run_name, config, seed, steps))
            (out_dir / "task_b.json").write_text(
                json.dumps([json.loads(r.model_dump_json()) for r in results])
            )

    # step 2: fold the runs into the summary, then write all three outputs.
    summaries, paired = summarize(results)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "summaries": [json.loads(s.model_dump_json()) for s in summaries],
                "paired": json.loads(paired.model_dump_json()),
            }
        )
    )
    (out_dir / "summary.md").write_text(_summary_md(summaries, paired))
    _write_figures(out_dir, config, seeds[0])


def main() -> None:
    # step 1: the smoke mode runs one seed for SMOKE_STEPS on the small config;
    # the full mode runs the asked seed count for the asked step count.
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        config = SpinGnnConfig(n=8, d=16, c=4, d_c=32, d_m=16, n_layers=2)
        run_ablation([0], SMOKE_STEPS, Path("results/smoke"), config)
    else:
        run_ablation(list(range(args.seeds)), args.steps, args.out, SpinGnnConfig())


if __name__ == "__main__":
    main()

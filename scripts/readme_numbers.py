# every number the README publishes, read from results/ so nothing is typed by hand.
# flow:
# 1. load task_b.json, summary.json, and summary.md from results/.
# 2. count tests, parameters, seeds, and the missing milestones.
# 3. decide the claims ledger rows C9 and C10 from the measured numbers.
# 4. print the frozen model as indented json to stdout.

import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from spin_gnn.constants import ACC_TARGET, N_SEEDS

_RUN_NAMES: tuple[str, ...] = ("full", "static", "scalar_only", "baseline")


def _tool(name: str) -> str:
    # step 1: prefer the repo .venv binary next to this script; fall back to PATH.
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / ".venv" / "bin" / name,
        Path(sys.executable).resolve().parent / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    assert found is not None, f"cannot find {name} in .venv or on PATH"
    return found


class ReadmeNumbers(BaseModel, frozen=True):
    # the frozen record the executor pastes from; every field is measured.
    commit: str
    date: str
    test_count: int
    params_full: int
    params_static: int
    params_scalar_only: int
    params_baseline: int
    heldout_full_mean: float
    heldout_full_ci: tuple[float, float]
    paired_diff_mean: float | None
    paired_diff_ci: tuple[float, float] | None
    paired_n: int
    milestone_missing_total: int
    sec_per_step_full: float
    max_step_after_full_seed0: float
    within_bound_after_full_seed0: bool
    c9_status: str
    c10_status: str
    summary_table: str


def _collect_test_count() -> int:
    # step 1: ask pytest how many tests the suite holds; do not run them.
    out = subprocess.run(
        [_tool("pytest"), "--collect-only", "-q"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in reversed(out.splitlines()):
        if "test" in line and "collected" in line:
            return int(line.split()[0])
    raise AssertionError("pytest collect-only printed no count line")


def _commit_short() -> str:
    # step 1: read the short hash of the checked-out commit.
    return subprocess.run(
        [_tool("git"), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def load_numbers(results_dir: Path) -> ReadmeNumbers:
    # requires: task_b.json, summary.json, summary.md exist under results_dir.
    task_b_path = results_dir / "task_b.json"
    summary_json_path = results_dir / "summary.json"
    summary_md_path = results_dir / "summary.md"
    assert task_b_path.exists(), f"missing {task_b_path}"
    assert summary_json_path.exists(), f"missing {summary_json_path}"
    assert summary_md_path.exists(), f"missing {summary_md_path}"

    records = json.loads(task_b_path.read_text())
    summary_json = json.loads(summary_json_path.read_text())
    summary_table = summary_md_path.read_text()

    # step 1: every run name covers the full seed set.
    by_name: dict[str, list[dict]] = {name: [] for name in _RUN_NAMES}
    for rec in records:
        by_name[rec["run_name"]].append(rec)
    for name in _RUN_NAMES:
        assert len(by_name[name]) >= N_SEEDS, (
            f"run {name} has {len(by_name[name])} records, fewer than N_SEEDS"
        )

    # step 2: per-config parameter counts come from seed 0 of each run.
    params = {
        name: int(next(r["params"] for r in by_name[name] if r["seed"] == 0))
        for name in _RUN_NAMES
    }

    # step 3: fold the full run into the headline numbers.
    summaries = {s["run_name"]: s for s in summary_json["summaries"]}
    full = summaries["full"]
    paired = summary_json["paired"]
    milestone_missing_total = int(sum(s["milestone_missing"] for s in summaries.values()))

    full_seed0 = next(r for r in by_name["full"] if r["seed"] == 0)
    max_step_after = float(full_seed0["max_step_after"])
    within_bound_after = bool(full_seed0["within_bound_after"])

    # step 4: close C9 and C10 per behavior rules 3 and 4.
    full_mean = float(full["heldout_mean"])
    full_ci = (float(full["heldout_ci"][0]), float(full["heldout_ci"][1]))
    c9_status = (
        "verified" if full_mean >= ACC_TARGET else f"failed; measured {full_mean:.4f}"
    )
    paired_n = int(paired["n_pairs"])
    paired_mean = paired["diff_mean"]
    paired_ci = paired["diff_ci"]
    if paired_n > 0 and paired_ci is not None:
        lo, hi = float(paired_ci[0]), float(paired_ci[1])
        excludes_zero = (lo > 0.0) or (hi < 0.0)
        c10_status = (
            "verified"
            if excludes_zero
            else f"not shown; paired interval [{lo:.4f}, {hi:.4f}] crosses zero"
        )
    else:
        c10_status = (
            "not shown; no paired seeds reached the milestone, interval undefined"
        )

    run_date = date.fromtimestamp(task_b_path.stat().st_mtime).isoformat()

    return ReadmeNumbers(
        commit=_commit_short(),
        date=run_date,
        test_count=_collect_test_count(),
        params_full=params["full"],
        params_static=params["static"],
        params_scalar_only=params["scalar_only"],
        params_baseline=params["baseline"],
        heldout_full_mean=full_mean,
        heldout_full_ci=full_ci,
        paired_diff_mean=None if paired_mean is None else float(paired_mean),
        paired_diff_ci=None if paired_ci is None else (float(paired_ci[0]), float(paired_ci[1])),
        paired_n=paired_n,
        milestone_missing_total=milestone_missing_total,
        sec_per_step_full=float(full["sec_per_step_mean"]),
        max_step_after_full_seed0=max_step_after,
        within_bound_after_full_seed0=within_bound_after,
        c9_status=c9_status,
        c10_status=c10_status,
        summary_table=summary_table,
    )


def main() -> None:
    # flow: load, count, decide, print.
    numbers = load_numbers(Path("results"))
    print(numbers.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

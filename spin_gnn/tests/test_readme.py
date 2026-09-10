# the readme test: the published numbers and status rows must match results/.
# spec: session-07 section 6. each test reads the files and checks one rule.

import importlib.util
import json
import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Protocol

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "readme_numbers.py"


def _load_readme_numbers() -> ModuleType:
    # the script lives outside the package; load it by path so pyright resolves.
    spec = importlib.util.spec_from_file_location("readme_numbers", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Numbers(Protocol):
    paired_n: int


_readme_numbers = _load_readme_numbers()
load_numbers: Callable[[Path], _Numbers] = _readme_numbers.load_numbers

from spin_gnn.constants import ACC_TARGET, N_SEEDS  # noqa: E402

_BANNED = (
    "delve",
    "leverage",
    "seamless",
    "robust",
    "comprehensive",
    "cutting-edge",
    "game-changing",
)


def _read(name: str) -> str:
    return (_REPO_ROOT / name).read_text()


def test_readme_has_no_placeholder() -> None:
    assert "__AFTER_TRAINING__" not in _read("README.md")


def test_math_has_no_placeholder() -> None:
    assert "__AFTER_TRAINING__" not in _read("docs/MATH.md")


def test_docs_have_no_em_dash() -> None:
    em_dash = chr(0x2014)
    for name in ("README.md", "docs/MATH.md", "docs/CLAIMS.md"):
        assert em_dash not in _read(name), f"{name} has an em-dash"


def test_readme_has_no_banned_words() -> None:
    text = _read("README.md")
    for word in _BANNED:
        assert not re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)


def test_readme_contains_the_summary_table_in_order() -> None:
    readme_lines = _read("README.md").splitlines()
    summary_lines = _read("results/summary.md").splitlines()
    cursor = 0
    for line in summary_lines:
        while cursor < len(readme_lines) and readme_lines[cursor] != line:
            cursor += 1
        assert cursor < len(readme_lines), f"missing summary line: {line!r}"
        cursor += 1


def test_claims_has_no_pending_row() -> None:
    # only table rows count; a row's status cell is its last pipe-delimited field.
    for line in _read("docs/CLAIMS.md").splitlines():
        if line.startswith("| C"):
            status = line.rsplit("|", 2)[-2].strip()
            assert status != "pending", f"row still pending: {line!r}"


def test_claims_c9_agrees_with_the_measured_mean() -> None:
    claims = _read("docs/CLAIMS.md")
    summary = json.loads(_read("results/summary.json"))
    full_mean = next(
        s["heldout_mean"] for s in summary["summaries"] if s["run_name"] == "full"
    )
    c9_row = next(line for line in claims.splitlines() if line.startswith("| C9 "))
    expect_verified = full_mean >= ACC_TARGET
    assert ("| verified |" in c9_row) == expect_verified
    if not expect_verified:
        assert "failed" in c9_row and f"{full_mean:.4f}" in c9_row


def test_claims_c10_agrees_with_the_paired_interval() -> None:
    claims = _read("docs/CLAIMS.md")
    summary = json.loads(_read("results/summary.json"))
    paired = summary["paired"]
    c10_row = next(line for line in claims.splitlines() if line.startswith("| C10 "))
    ci = paired["diff_ci"]
    excludes_zero = (
        paired["n_pairs"] > 0
        and ci is not None
        and (float(ci[0]) > 0.0 or float(ci[1]) < 0.0)
    )
    assert ("| verified |" in c10_row) == excludes_zero
    if not excludes_zero:
        assert "not shown" in c10_row


def test_load_numbers_runs_and_pairs_within_seed_budget() -> None:
    numbers = load_numbers(_REPO_ROOT / "results")
    assert numbers.paired_n <= N_SEEDS

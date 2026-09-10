# the cached Task B stream: the exact batch sequence one seed would draw, built
# once and shared by every run of that seed. spec: docs/SPEC.md.
# the constructor is pure python per element and costs about a third of a
# training step on CPU, so caching it removes that cost from three of the four
# runs per seed. the batches are bitwise the ones the uncached loop would draw.
# flow:
# 1. a cache key names the satellite count, the seed, the batch, and the steps.
# 2. load returns the stored stream when a file with at least that many steps
#    exists; otherwise the stream is drawn from the seeded generator and saved.
# 3. batches rebuild the zero state fields from the run config on the way out.

from pathlib import Path

import torch
from torch import Tensor

from spin_gnn.constants import CACHE_DIR
from spin_gnn.constellation import assert_valid
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.train.tasks_synthetic import construct_task_b_flat, label_task_b
from spin_gnn.types import Constellation

_FIELDS: tuple[str, ...] = ("x", "s", "u", "phi", "omega", "x_c")


def cache_root() -> Path:
    # step 1: the cache sits under results/ beside the run outputs.
    repo = Path(__file__).resolve().parents[2]
    return repo / CACHE_DIR


def _key(n: int, seed: int, batch: int, steps: int) -> str:
    return f"task_b_n{n}_seed{seed}_b{batch}_steps{steps}.pt"


def _find(n: int, seed: int, batch: int, steps: int, root: Path) -> Path | None:
    # step 1: any file for this (n, seed, batch) with at least steps batches serves.
    best: Path | None = None
    best_steps = -1
    for path in root.glob(f"task_b_n{n}_seed{seed}_b{batch}_steps*.pt"):
        stored = int(path.stem.rsplit("steps", 1)[1])
        if stored >= steps and (best is None or stored < best_steps):
            best, best_steps = path, stored
    return best


def _draw(
    n: int, seed: int, batch: int, steps: int, config: SpinGnnConfig
) -> dict[str, Tensor]:
    # step 1: the seeded generator draws every batch in order, as the loop would.
    generator = torch.Generator().manual_seed(seed)
    parts: dict[str, list[Tensor]] = {name: [] for name in (*_FIELDS, "labels")}
    for _ in range(steps):
        c, labels = construct_task_b_flat(batch, config, generator)
        for name in _FIELDS:
            parts[name].append(getattr(c, name))
        parts["labels"].append(labels)
    stacked = {name: torch.stack(values) for name, values in parts.items()}
    assert stacked["x"].shape == (steps, batch, n, 3)
    assert stacked["labels"].shape == (steps, batch)
    return stacked


def _rebuild(stored: dict[str, Tensor], index: int, config: SpinGnnConfig) -> Constellation:
    # step 1: one batch from the stack, with the zero state fields at run width.
    x = stored["x"][index]
    b, n = int(x.shape[0]), int(x.shape[1])
    out = Constellation(
        x=x,
        s=stored["s"][index],
        u=stored["u"][index],
        phi=stored["phi"][index],
        omega=stored["omega"][index],
        h=torch.zeros(b, n, config.d, dtype=x.dtype),
        v=torch.zeros(b, n, 3, config.c, dtype=x.dtype),
        x_c=stored["x_c"][index],
        h_c=torch.zeros(b, config.d_c, dtype=x.dtype),
    )
    return out


class TaskBStream:
    # the batch sequence for one seed; index it by step (0-based).
    def __init__(
        self,
        seed: int,
        steps: int,
        batch: int,
        config: SpinGnnConfig,
        root: Path | None = None,
        use_cache: bool = True,
    ) -> None:
        assert steps >= 1 and batch >= 1, "steps and batch must be positive"
        self.seed: int = seed
        self.steps: int = steps
        self.batch: int = batch
        self.config: SpinGnnConfig = config
        n = config.n
        root = cache_root() if root is None else root
        found = _find(n, seed, batch, steps, root) if use_cache else None
        if found is not None:
            stored = torch.load(found, weights_only=True)
            self.stored: dict[str, Tensor] = {k: v[:steps] for k, v in stored.items()}
            self.path: Path | None = found
        else:
            self.stored = _draw(n, seed, batch, steps, config)
            self.path = None
            if use_cache:
                root.mkdir(parents=True, exist_ok=True)
                self.path = root / _key(n, seed, batch, steps)
                torch.save(self.stored, self.path)
        assert self.stored["labels"].shape == (steps, batch)

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, index: int) -> tuple[Constellation, Tensor]:
        # step 1: rebuild the batch and check its labels against the labeler.
        assert 0 <= index < self.steps, f"index {index} outside [0, {self.steps})"
        c = _rebuild(self.stored, index, self.config)
        labels = self.stored["labels"][index]
        assert_valid(c)
        return c, labels

    def verify(self, index: int) -> bool:
        # step 1: the stored labels must match the labeler on the rebuilt batch.
        c, labels = self[index]
        return bool((label_task_b(c) == labels).all())

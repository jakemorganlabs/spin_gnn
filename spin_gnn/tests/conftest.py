# shared pytest and Hypothesis plumbing for the geometry suite.
# flow:
# 1. register and load a strict Hypothesis profile so properties run deterministically.
# 2. provide a seeded torch generator fixture.
# 3. provide numpy strategies for box points, unit vectors, and angles.

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

from spin_gnn.constants import BOX, SEED

settings.register_profile(
    "spin_gnn",
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=list(HealthCheck),
)
settings.load_profile("spin_gnn")


@pytest.fixture()
def generator() -> torch.Generator:
    # one seeded generator per test so randomness is reproducible.
    return torch.Generator().manual_seed(SEED)


def box_points(shape: tuple[int, ...] = ()) -> st.SearchStrategy[np.ndarray]:
    # points strictly inside the box so reflection properties stay in-contract.
    # margin 0.2 keeps the point plus a +0.2 step inside the box.
    w, h, d = BOX
    margin = 0.2
    return st.tuples(
        st.floats(margin, w - margin),
        st.floats(margin, h - margin),
        st.floats(margin, d - margin),
    ).map(lambda p: np.broadcast_to(np.asarray(p, dtype=np.float64), shape + (3,)).copy())


def free_points(
    shape: tuple[int, ...] = (), low: float = -3.0, high: float = 3.0
) -> st.SearchStrategy:
    # arbitrary points in R^3 for projection properties.
    return st.tuples(
        st.floats(low, high),
        st.floats(low, high),
        st.floats(low, high),
    ).map(lambda p: np.broadcast_to(np.asarray(p, dtype=np.float64), shape + (3,)).copy())


def displacements(limit: float = 1.9, shape: tuple[int, ...] = ()) -> st.SearchStrategy:
    # per-axis displacement below two box extents, as the reflection contract needs.
    return st.tuples(
        st.floats(-limit, limit),
        st.floats(-limit, limit),
        st.floats(-limit, limit),
    ).map(lambda p: np.broadcast_to(np.asarray(p, dtype=np.float64), shape + (3,)).copy())


def unit_vectors(shape: tuple[int, ...] = ()) -> st.SearchStrategy:
    # unit 3-vectors drawn as normalized gaussians, away from zero norm.

    def _make(seed_frac: tuple[float, float, float]) -> np.ndarray:
        v = np.asarray(seed_frac, dtype=np.float64)
        v = v * 2 - 1
        norm = float(np.sqrt((v * v).sum()))
        if norm < 1e-3:
            v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            norm = 1.0
        unit = (v / norm).astype(np.float64)
        return np.broadcast_to(unit, shape + (3,)).copy()

    return st.tuples(
        st.floats(0.0, 1.0),
        st.floats(0.0, 1.0),
        st.floats(0.0, 1.0),
    ).map(_make)


def angles(low: float = -100.0, high: float = 100.0) -> st.SearchStrategy:
    # free angle draws across many turns so wrapping is exercised.
    return st.floats(low, high, allow_nan=False, allow_infinity=False)

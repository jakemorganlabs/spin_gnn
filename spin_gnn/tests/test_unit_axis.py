# examples and properties for normalize and renormalize_axis.
# flow:
# 1. pin the normalize examples.
# 2. pin the normalize and renormalize_axis properties.

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st

from spin_gnn.constants import SEED
from spin_gnn.geometry.box import normalize, safe_norm
from spin_gnn.geometry.spin import renormalize_axis
from spin_gnn.layers.message import MessageLayer
from spin_gnn.layers.update import SatelliteUpdate
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.tests.conftest import unit_vectors
from spin_gnn.tests.test_constellation import make_constellation

F64 = torch.float64


# example: a known axis normalizes to the unit axis.


def test_normalize_scales_axis_to_unit() -> None:
    v = torch.tensor([2.0, 0.0, 0.0], dtype=F64)
    out = normalize(v)
    assert torch.allclose(out, torch.tensor([1.0, 0.0, 0.0], dtype=F64), atol=1e-8)


def test_renormalize_axis_makes_norm_three_vector_unit() -> None:
    v = torch.tensor([3.0, 0.0, 0.0], dtype=F64)
    out = renormalize_axis(v)
    assert torch.allclose(safe_norm(out), torch.tensor(1.0, dtype=F64), atol=1e-9)


def test_update_axis_output_is_unit_norm(generator: torch.Generator) -> None:
    # the axis update emits exactly unit axes on a random interior fixture.
    torch.manual_seed(SEED)
    config = SpinGnnConfig(n=6, d=16, c=4, d_c=32, d_m=16)
    layer = MessageLayer(config).double()
    update = SatelliteUpdate(config).double()
    c = make_constellation(3, 6, 16, 4, generator, interior=True)
    out = update(c, layer(c))
    norms = out.u.norm(dim=-1)
    gap = (norms - 1.0).abs()
    assert bool((gap <= 1e-5).all()), f"axis norm off by {float(gap.max())}"


# property: a scaled unit vector normalizes to unit and is idempotent.


@given(u=unit_vectors(), scale=st.floats(1e-1, 10.0, allow_nan=False, allow_infinity=False))
def test_normalize_is_unit_and_idempotent(u: np.ndarray, scale: float) -> None:
    v = torch.from_numpy(np.ascontiguousarray(u)).to(F64) * scale
    out = normalize(v)
    # safe_norm adds eps inside the sqrt, so unit is met to the documented 1e-5.
    assert torch.allclose(safe_norm(out), torch.tensor(1.0, dtype=F64), atol=1e-5)
    assert torch.allclose(normalize(out), out, atol=1e-9)

# examples and property for the IdentityEncoder. spec: docs/SPEC.md.
# flow:
# 1. the example pins zero V, the h shape, and the broadcast h_c.
# 2. the property proves h invariant under Haar rotations on the interior.

import torch
from hypothesis import given
from hypothesis import strategies as st

from spin_gnn.constants import SEED
from spin_gnn.constellation import rotate_about_controller
from spin_gnn.encode.identity_encoder import IdentityEncoder
from spin_gnn.geometry.frames import random_rotation
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.tests.test_constellation import make_constellation


def test_encoder_seeds_zero_v_and_shapes(generator: torch.Generator) -> None:
    torch.manual_seed(SEED)
    d, c, d_c = 16, 4, 32
    config = SpinGnnConfig(n=6, d=d, c=c, d_c=d_c, d_m=16)
    raw = make_constellation(3, 6, d, c, generator, interior=True)
    out = IdentityEncoder(config).double()(raw)
    assert bool((out.v == 0).all())
    assert out.v.shape == (3, 6, 3, c)
    assert out.h.shape == (3, 6, d)
    assert out.h_c.shape == (3, d_c)
    # the controller state is one learned constant broadcast over the batch.
    assert bool((out.h_c == out.h_c[0:1]).all())
    # the raw geometric fields are the identical tensors.
    assert out.x is raw.x
    assert out.s is raw.s
    assert out.u is raw.u
    assert out.phi is raw.phi
    assert out.omega is raw.omega
    assert out.x_c is raw.x_c


@given(st.integers(min_value=1, max_value=3))
def test_encoder_h_invariant_under_rotation(seed_offset: int) -> None:
    gen = torch.Generator().manual_seed(SEED + 41 + seed_offset)
    torch.manual_seed(SEED)
    d, c = 16, 4
    config = SpinGnnConfig(n=6, d=d, c=c, d_c=32, d_m=16)
    encoder = IdentityEncoder(config).double()
    raw = make_constellation(3, 6, d, c, gen, interior=True)
    rotation = random_rotation(3, gen)
    before = encoder(raw).h
    after = encoder(rotate_about_controller(raw, rotation)).h
    gap = (before - after).abs()
    assert bool((gap <= 1e-9).all()), f"h moved by {float(gap.max())}"

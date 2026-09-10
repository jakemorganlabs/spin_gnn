# examples and properties for the session 8 pieces: the gaussian basis, the
# attention weights, the signed speed clamp, the learning-rate schedule, and
# the cached Task B stream. spec: docs/SPEC.md, Prop 9 in docs/MATH.md.
# flow:
# 1. the basis has the stated shape, lies in (0, 1], and peaks at its centers.
# 2. the attention weights form a distribution over N neighbors with a zero
#    diagonal, and they are invariant under rotation.
# 3. clamp_speed keeps the sign and folds into [-OMEGA_MAX, OMEGA_MAX].
# 4. the schedule warms up linearly, peaks at 1, and ends at LR_MIN_FRAC.
# 5. the stream draws bitwise the batches the uncached loop would draw, and a
#    second load from disk returns the same tensors.

import math

import torch
from hypothesis import given
from hypothesis import strategies as st

from spin_gnn.constants import H_C_INIT_STD, LR_MIN_FRAC, OMEGA_MAX, SEED, WARMUP_STEPS
from spin_gnn.constellation import rotate_about_controller
from spin_gnn.encode.identity_encoder import IdentityEncoder
from spin_gnn.geometry.basis import gaussian_rbf
from spin_gnn.geometry.frames import random_rotation
from spin_gnn.geometry.spin import clamp_speed
from spin_gnn.layers.message import MessageLayer, feature_width, featurize
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.tests.test_constellation import make_constellation
from spin_gnn.train.loop import lr_multiplier
from spin_gnn.train.stream import TaskBStream
from spin_gnn.train.tasks_synthetic import construct_task_b_flat


def _small() -> SpinGnnConfig:
    return SpinGnnConfig(n=6, d=16, c=4, d_c=32, d_m=16)


def test_rbf_shape_range_and_peaks() -> None:
    x = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    out = gaussian_rbf(x, 0.0, 1.0, 5)
    assert out.shape == (5, 5)
    assert bool(((out > 0.0) & (out <= 1.0)).all())
    # each input sits on its own center, so the diagonal is exactly one and
    # every off-diagonal bump is strictly smaller.
    assert torch.allclose(out.diagonal(), torch.ones(5, dtype=torch.float64))
    off = out - torch.eye(5, dtype=torch.float64)
    assert bool((off < 1.0).all())


def test_rbf_neighbors_overlap_at_half_height() -> None:
    # the width equals the spacing, so a center reads exp(-1/2) at its neighbor.
    x = torch.tensor([0.25], dtype=torch.float64)
    out = gaussian_rbf(x, 0.0, 1.0, 5)[0]
    assert abs(float(out[1]) - 1.0) <= 1e-12
    assert abs(float(out[0]) - math.exp(-0.5)) <= 1e-12


def test_feature_width_counts_both_grids() -> None:
    config = _small()
    c = make_constellation(2, config.n, config.d, config.c, torch.Generator().manual_seed(SEED))
    from spin_gnn.geometry.invariants import edge_packet

    feats = featurize(edge_packet(c))
    assert feats.shape == (2, config.n, config.n, feature_width(config.d))


def test_attention_is_a_distribution_with_zero_diagonal(generator: torch.Generator) -> None:
    torch.manual_seed(SEED)
    config = _small()
    layer = MessageLayer(config).double()
    c = make_constellation(3, config.n, config.d, config.c, generator, interior=True)
    attn = layer(c).attn
    assert attn.shape == (3, config.n, config.n + 1)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(3, config.n, dtype=torch.float64))
    assert bool((attn[..., : config.n].diagonal(dim1=1, dim2=2) == 0).all())
    assert bool((attn >= 0).all())


@given(st.integers(min_value=1, max_value=3))
def test_attention_invariant_under_rotation(seed_offset: int) -> None:
    gen = torch.Generator().manual_seed(SEED + 211 + seed_offset)
    torch.manual_seed(SEED)
    config = _small()
    layer = MessageLayer(config).double()
    c = make_constellation(3, config.n, config.d, config.c, gen, interior=True)
    rotation = random_rotation(3, gen)
    before = layer(c).attn
    after = layer(rotate_about_controller(c, rotation)).attn
    gap = (before - after).abs()
    assert bool((gap <= 1e-9).all()), f"attention moved by {float(gap.max())}"


@given(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
def test_clamp_speed_keeps_sign_and_range(value: float) -> None:
    out = float(clamp_speed(torch.tensor(value, dtype=torch.float64)))
    assert -OMEGA_MAX <= out <= OMEGA_MAX
    if abs(value) <= OMEGA_MAX:
        assert out == value
    else:
        assert out == (OMEGA_MAX if value > 0 else -OMEGA_MAX)


def test_lr_schedule_warms_up_peaks_and_decays() -> None:
    total = 4000
    warmup = min(WARMUP_STEPS, total // 10)
    assert lr_multiplier(0, total) == 1.0 / warmup
    assert abs(lr_multiplier(warmup - 1, total) - 1.0) <= 1e-12
    assert abs(lr_multiplier(warmup, total) - 1.0) <= 1e-12
    assert abs(lr_multiplier(total - 1, total) - LR_MIN_FRAC) <= 1e-6
    values = [lr_multiplier(step, total) for step in range(warmup, total)]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(values, values[1:], strict=False))


def test_lr_schedule_short_runs_stay_bounded() -> None:
    for total in (1, 2, 5, 20):
        for step in range(total):
            value = lr_multiplier(step, total)
            assert 0.0 < value <= 1.0


def test_stream_matches_uncached_draw_and_reloads(tmp_path) -> None:
    config = SpinGnnConfig(n=8, d=16, c=4, d_c=32, d_m=16, n_layers=2)
    steps, batch, seed = 3, 8, 5
    stream = TaskBStream(seed, steps, batch, config, root=tmp_path)
    generator = torch.Generator().manual_seed(seed)
    for index in range(steps):
        expected_c, expected_labels = construct_task_b_flat(batch, config, generator)
        c, labels = stream[index]
        assert torch.equal(c.x, expected_c.x)
        assert torch.equal(c.u, expected_c.u)
        assert torch.equal(c.omega, expected_c.omega)
        assert torch.equal(labels, expected_labels)
        assert stream.verify(index)
    # a second construction reads the file and returns the same tensors.
    assert stream.path is not None and stream.path.exists()
    again = TaskBStream(seed, steps, batch, config, root=tmp_path)
    assert again.path == stream.path
    for index in range(steps):
        assert torch.equal(again[index][0].x, stream[index][0].x)
    # a shorter request reuses the longer file by slicing.
    shorter = TaskBStream(seed, 2, batch, config, root=tmp_path)
    assert shorter.path == stream.path
    assert len(shorter) == 2
    assert torch.equal(shorter[1][0].x, stream[1][0].x)


def test_controller_seed_starts_at_unit_scale() -> None:
    # a layer norm reads the seed in the first controller update; its
    # Jacobian scales as one over the seed spread, so the spread must be O(1).
    torch.manual_seed(SEED)
    encoder = IdentityEncoder(SpinGnnConfig())
    spread = float(encoder.h_c_param.std())
    assert 0.5 * H_C_INIT_STD <= spread <= 1.5 * H_C_INIT_STD

"""Prop 2, 3, 4."""

# flow:
# 1. fixtures: interior_batch, wall_touching_batch, enlarge_position_step, small_config.
# 2. examples: octahedral exactness, exactness under an enlarged position
#    step, the generic-rotation break at the walls, and the generic hold inside.
# 3. the max-step example measures the per-layer bound at init.
# 4. properties: Prop 3 invariance and equivariance, Prop 4 everywhere.

import torch
from hypothesis import given
from hypothesis import strategies as st
from torch import Tensor

from spin_gnn.constellation import (
    assert_valid,
    in_interior,
    rotate_about_controller,
)
from spin_gnn.geometry.frames import (
    octahedral_rotations,
    random_rotation,
    rotate_channels,
    rotate_vectors,
)
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.model.diagnostics import measure_max_step, symmetry_gap
from spin_gnn.model.spin_gnn_gnn import SpinGnn, SpinGnnLayer, build_model
from spin_gnn.tests.test_constellation import make_constellation
from spin_gnn.types import Constellation

TOL: float = 1e-8
# the break floor sits four orders above the interior tolerance TOL, so a gap
# past it is a symmetry break and not float noise; the layer norms of the
# session 8 model damp the h_c break to a few 1e-4 at init.
_BREAK_FLOOR: float = 1e-4

WALL_CENTERS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.5, 0.5),
    (0.0, 0.5, 0.5),
    (0.5, 1.0, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 1.0),
    (0.5, 0.5, 0.0),
)


def small_config(**overrides: bool) -> SpinGnnConfig:
    config = SpinGnnConfig(n=8, d=16, c=4, d_c=32, d_m=16, n_layers=2)
    if overrides:
        return config.model_copy(update=dict(overrides))
    return config


def interior_batch(b: int, config: SpinGnnConfig, generator: torch.Generator) -> Constellation:
    # x_c pinned at the center so every rotation about it has a fixed point.
    c = make_constellation(b, config.n, config.d, config.c, generator, interior=True)
    return _edited(c, x_c=torch.full((b, 3), 0.5, dtype=c.x.dtype))


def wall_touching_batch(config: SpinGnnConfig, generator: torch.Generator) -> Constellation:
    # six satellites pinned at the wall centers; every satellite sits within 0.5
    # of the center, so any rotation about the center stays inside the box.
    c = make_constellation(1, config.n, config.d, config.c, generator, interior=True)
    x_c = torch.full((1, 3), 0.5, dtype=c.x.dtype)
    moved = c.x.clone()
    for k, point in enumerate(WALL_CENTERS):
        moved[0, k] = torch.tensor(point, dtype=c.x.dtype)
    radii = (moved - x_c.unsqueeze(1)).norm(dim=-1)
    assert bool((radii <= 0.5 + 1e-12).all()), "every satellite must stay within 0.5"
    out = _edited(c, x=moved, x_c=x_c)
    assert_valid(out)
    return out


def enlarge_position_step(model: SpinGnn) -> None:
    # the near-zero init hides the clipping effect; set the final layer of every
    # phi_x to a weight of 1.0 so the wall break is measurable. deterministic.
    for layer in model.layers:
        assert isinstance(layer, SpinGnnLayer), "model.layers must hold SpinGnnLayer"
        phi_x = layer.update.phi_x
        last = phi_x[-1]
        assert isinstance(last, torch.nn.Linear), "phi_x must end in a Linear"
        with torch.no_grad():
            last.weight.fill_(1.0)
            last.bias.fill_(0.0)


def _edited(c: Constellation, **fields: Tensor) -> Constellation:
    data = dict(x=c.x, s=c.s, u=c.u, phi=c.phi, omega=c.omega, h=c.h, v=c.v, x_c=c.x_c, h_c=c.h_c)
    data.update(fields)
    return Constellation(**data)  # type: ignore[arg-type]


def _assert_equivariant(
    model: SpinGnn, c: Constellation, rotation: Tensor, tol: float = TOL
) -> None:
    before = model(c).constellation
    after = model(rotate_about_controller(c, rotation)).constellation
    rel_gap = (after.x - after.x_c.unsqueeze(1)) - rotate_vectors(
        rotation, before.x - before.x_c.unsqueeze(1)
    )
    assert bool((rel_gap.abs() <= tol).all()), (
        f"x - x_c not equivariant by {float(rel_gap.abs().max())}"
    )
    u_gap = after.u - rotate_vectors(rotation, before.u)
    assert bool((u_gap.abs() <= tol).all()), f"u not equivariant by {float(u_gap.abs().max())}"
    v_gap = after.v - rotate_channels(rotation, before.v)
    assert bool((v_gap.abs() <= tol).all()), f"v not equivariant by {float(v_gap.abs().max())}"
    for name in ("h_c", "h", "s", "phi", "omega"):
        gap = (getattr(after, name) - getattr(before, name)).abs()
        assert bool((gap <= tol).all()), f"{name} moved by {float(gap.max())}"


def test_octahedral_exact_on_wall_touching_batch(generator: torch.Generator) -> None:
    # Prop 4: the model is exactly O-equivariant everywhere, float64, all 24.
    model = build_model(small_config()).double()
    c = wall_touching_batch(small_config(), generator)
    for k in range(24):
        rotation = octahedral_rotations()[k].unsqueeze(0).to(c.x.dtype)
        _assert_equivariant(model, c, rotation)


def test_octahedral_exact_survives_enlarged_position_step(
    generator: torch.Generator,
) -> None:
    # the cube symmetry survives even with the position step inflated.
    config = small_config()
    model = build_model(config).double()
    enlarge_position_step(model)
    c = wall_touching_batch(config, generator)
    for k in range(24):
        rotation = octahedral_rotations()[k].unsqueeze(0).to(c.x.dtype)
        _assert_equivariant(model, c, rotation)


def test_generic_breaks(generator: torch.Generator) -> None:
    # Prop 5 (C5): with the step inflated, a generic rotation of a wall-touching
    # constellation moves h_c by more than 1e-3 because the box clips anisotropically.
    # the effect is small at one layer, so the gap is the maximum over a handful
    # of Haar draws; the cube symmetry would pin every one of them to zero.
    config = small_config()
    model = build_model(config).double()
    enlarge_position_step(model)
    c = wall_touching_batch(config, generator)
    worst = 0.0
    for _ in range(4):
        rotation = random_rotation(1, generator)
        gap = float(symmetry_gap(model, c, rotation).max())
        worst = max(worst, gap)
    print(f"\ngeneric-breaks gap: {worst}")
    assert worst > _BREAK_FLOOR


def test_generic_rotation_holds_on_interior_after_enlarge(
    generator: torch.Generator,
) -> None:
    # the same inflated model is still equivariant inside U: the walls are far.
    config = small_config()
    model = build_model(config).double()
    enlarge_position_step(model)
    c = interior_batch(2, config, generator)
    rotation = random_rotation(2, generator)
    gap = symmetry_gap(model, c, rotation)
    assert bool((gap <= TOL).all()), f"interior gap {float(gap.max())}"


def test_measure_max_step_within_bound_at_init(generator: torch.Generator) -> None:
    config = SpinGnnConfig()
    model = build_model(config).double()
    gen = torch.Generator().manual_seed(7331)
    batches = [interior_batch(16, config, gen) for _ in range(8)]
    report = measure_max_step(model, batches)
    print(f"\nmax position step per layer: {report.max_step:.6f} (bound {report.bound:.6f})")
    assert report.within_bound
    assert len(report.per_layer) == config.n_layers
    assert all(0.0 <= value <= report.max_step for value in report.per_layer)


@given(st.integers(min_value=1, max_value=3))
def test_model_invariant_h_c_y_z_on_interior(seed_offset: int) -> None:
    # Prop 3: the invariant outputs stay put under Haar rotations inside U.
    config = small_config()
    model = build_model(config).double()
    gen = torch.Generator().manual_seed(9301 + seed_offset)
    c = interior_batch(2, config, gen)
    rotation = random_rotation(2, gen)
    before = model(c)
    after = model(rotate_about_controller(c, rotation))
    for name, tensor_a, tensor_b in (
        ("h_c", before.constellation.h_c, after.constellation.h_c),
        ("y", before.y, after.y),
        ("z", before.z, after.z),
    ):
        gap = (tensor_a - tensor_b).abs()
        assert bool((gap <= TOL).all()), f"{name} moved by {float(gap.max())}"


@given(st.integers(min_value=1, max_value=3))
def test_model_equivariant_fields_on_interior(seed_offset: int) -> None:
    # Prop 3: x - x_c, u, and v transform by the same rotation inside U.
    config = small_config()
    model = build_model(config).double()
    gen = torch.Generator().manual_seed(9413 + seed_offset)
    c = interior_batch(2, config, gen)
    rotation = random_rotation(2, gen)
    _assert_equivariant(model, c, rotation)


@given(st.integers(min_value=1, max_value=3))
def test_octahedral_equivalence_everywhere(seed_offset: int) -> None:
    # Prop 4: all 24 octahedral rotations commute exactly, valid c or not.
    config = small_config()
    model = build_model(config).double()
    gen = torch.Generator().manual_seed(9527 + seed_offset)
    c = make_constellation(1, config.n, config.d, config.c, gen, interior=False)
    c = _edited(c, x_c=torch.full((1, 3), 0.5, dtype=c.x.dtype))
    assert_valid(c)
    for k in range(24):
        rotation = octahedral_rotations()[k].unsqueeze(0).to(c.x.dtype)
        rel_gap = (
            model(rotate_about_controller(c, rotation)).constellation.h_c
            - model(c).constellation.h_c
        ).abs()
        assert bool((rel_gap <= TOL).all()), (
            f"h_c moved by {float(rel_gap.max())} on octahedral index {k}"
        )


@given(st.integers(min_value=1, max_value=3))
def test_symmetry_gap_below_tol_on_interior(seed_offset: int) -> None:
    # Prop 3 helper: symmetry_gap itself stays under 1e-8 on interior Haar draws.
    config = small_config()
    model = build_model(config).double()
    gen = torch.Generator().manual_seed(9637 + seed_offset)
    c = interior_batch(2, config, gen)
    rotation = random_rotation(2, gen)
    gap = symmetry_gap(model, c, rotation)
    assert bool((gap <= TOL).all()), f"gap {float(gap.max())}"


@given(st.integers(min_value=1, max_value=2))
def test_interior_batch_fixture_stays_interior(seed_offset: int) -> None:
    gen = torch.Generator().manual_seed(9741 + seed_offset)
    c = interior_batch(2, small_config(), gen)
    assert bool(in_interior(c).all())

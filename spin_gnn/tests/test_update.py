# examples and properties for the node updates, the pool, and the controller.
# flow:
# 1. examples pin validation, flag behavior, the phase integrator, and the
#    step bound at init.
# 2. properties prove the pool invariant, each update equivariant or invariant,
#    the output valid, and the identical-tensor rule per flag.

import torch
from hypothesis import given
from hypothesis import strategies as st

from spin_gnn.constants import DT, MARGIN, N_LAYERS, OMEGA_MAX, SEED
from spin_gnn.constellation import rotate_about_controller, validate
from spin_gnn.geometry.frames import random_rotation, rotate_channels, rotate_vectors
from spin_gnn.geometry.spin import wrap_angle
from spin_gnn.layers.controller import ControllerPool
from spin_gnn.layers.message import MessageLayer
from spin_gnn.layers.update import ControllerUpdate, SatelliteUpdate
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.tests.test_constellation import make_constellation
from spin_gnn.types import Constellation, ValidOutcome

_SMALL = dict(n=6, d=16, c=4, d_c=32, d_m=16)


def _config(**flags: bool) -> SpinGnnConfig:
    return SpinGnnConfig(**_SMALL, **flags)


def _seeded(cls: type, config: SpinGnnConfig):
    torch.manual_seed(SEED)
    return cls(config).double()


def _msg_and_c(gen: torch.Generator, config: SpinGnnConfig, batch: int = 3):
    c = make_constellation(batch, config.n, config.d, config.c, gen, interior=True)
    msg = _seeded(MessageLayer, config)(c)
    return c, msg


def _with_fields(c: Constellation, **fields: torch.Tensor) -> Constellation:
    data = dict(x=c.x, s=c.s, u=c.u, phi=c.phi, omega=c.omega, h=c.h, v=c.v, x_c=c.x_c, h_c=c.h_c)
    data.update(fields)
    return Constellation(**data)  # type: ignore[arg-type]


def test_update_output_validates(generator: torch.Generator) -> None:
    config = _config()
    c, msg = _msg_and_c(generator, config)
    out = _seeded(SatelliteUpdate, config)(c, msg)
    assert isinstance(validate(out), ValidOutcome)


def test_update_static_returns_identical_tensors(generator: torch.Generator) -> None:
    # a pure static run freezes every geometric field; with the learned speed off
    # too, the identical-tensor rule holds for all six fields.
    flags = dict(
        update_size=False,
        update_axis=False,
        update_speed=False,
        update_phase=False,
        update_position=False,
        use_vectors=False,
    )
    config = _config(**flags)
    c, msg = _msg_and_c(generator, config)
    out = _seeded(SatelliteUpdate, config)(c, msg)
    assert out.x is c.x
    assert out.u is c.u
    assert out.s is c.s
    assert out.omega is c.omega
    assert out.phi is c.phi
    assert out.v is c.v
    # the scalar update always runs, so h must differ.
    assert not bool((out.h == c.h).all())


def test_update_phase_integrator_when_speed_frozen(generator: torch.Generator) -> None:
    config = _config(update_speed=False, update_phase=True)
    c, msg = _msg_and_c(generator, config)
    full_omega = torch.full_like(c.omega, OMEGA_MAX)
    zero_phi = torch.zeros_like(c.phi)
    c = _with_fields(c, omega=full_omega, phi=zero_phi)
    out = _seeded(SatelliteUpdate, config)(c, msg)
    expected = wrap_angle(zero_phi + full_omega * DT)
    gap = (out.phi - expected).abs()
    assert bool((gap <= 1e-9).all()), f"phi moved by {float(gap.max())}"


def test_update_step_bound_at_init(generator: torch.Generator) -> None:
    # phi_x's final layer is scaled to INIT_SMALL so the position step is O(1e-3),
    # which provably sits below MARGIN / N_LAYERS at init. Rounding down to an
    # integer pixels keeps the assertion exact rather than approximate.
    config = _config()
    c, msg = _msg_and_c(generator, config, batch=32)
    update = _seeded(SatelliteUpdate, config)
    from spin_gnn.constants import INIT_SMALL

    with torch.no_grad():
        update.phi_x[-1].weight.mul_(INIT_SMALL)
        update.phi_x[-1].bias.mul_(INIT_SMALL)
    out = update(c, msg)
    max_step = float(torch.linalg.norm((out.x - c.x).detach().float(), dim=-1).max())
    bound = MARGIN / N_LAYERS
    print(f"\nmax ||dx|| at init: {max_step:.3e}  bound: {bound:.3e}")
    assert max_step < bound, f"step {max_step} must stay below the bound {bound}"


def test_pool_shape(generator: torch.Generator) -> None:
    config = _config()
    c = make_constellation(3, config.n, config.d, config.c, generator, interior=True)
    pool = _seeded(ControllerPool, config)(c)
    assert pool.shape == (3, 3 * 16 + 3)


def test_controller_update_shape(generator: torch.Generator) -> None:
    config = _config()
    c, msg = _msg_and_c(generator, config)
    pool = _seeded(ControllerPool, config)(c)
    out = _seeded(ControllerUpdate, config)(c.h_c, msg.m_c, pool)
    assert out.shape == (3, 32)
    assert not bool((out == c.h_c).all())


@given(st.integers(min_value=1, max_value=3))
def test_pool_invariant_under_rotation(seed_offset: int) -> None:
    gen = torch.Generator().manual_seed(SEED + 211 + seed_offset)
    config = _config()
    pool_mod = _seeded(ControllerPool, config)
    c = make_constellation(3, config.n, config.d, config.c, gen, interior=True)
    rotation = random_rotation(3, gen)
    before = pool_mod(c)
    after = pool_mod(rotate_about_controller(c, rotation))
    gap = (before - after).abs()
    assert bool((gap <= 1e-9).all()), f"pool moved by {float(gap.max())}"


@given(st.integers(min_value=1, max_value=3))
def test_update_equivariance(seed_offset: int) -> None:
    # per-update: h, s, omega, phi invariant; x, u, v equivariant, on interior.
    gen = torch.Generator().manual_seed(SEED + 307 + seed_offset)
    config = _config()
    layer = _seeded(MessageLayer, config)
    update = _seeded(SatelliteUpdate, config)
    c = make_constellation(3, config.n, config.d, config.c, gen, interior=True)
    rotation = random_rotation(3, gen)
    before = update(c, layer(c))
    rotated_c = rotate_about_controller(c, rotation)
    after = update(rotated_c, layer(rotated_c))
    for name in ("h", "s", "omega", "phi"):
        gap = (getattr(before, name) - getattr(after, name)).abs()
        assert bool((gap <= 1e-8).all()), f"{name} moved by {float(gap.max())}"
    expected_x = rotate_vectors(rotation, before.x - c.x_c.unsqueeze(1)) + c.x_c.unsqueeze(1)
    gap = (expected_x - after.x).abs()
    assert bool((gap <= 1e-8).all()), f"x moved by {float(gap.max())}"
    gap = (rotate_vectors(rotation, before.u) - after.u).abs()
    assert bool((gap <= 1e-8).all()), f"u moved by {float(gap.detach().max())}"
    gap = (rotate_channels(rotation, before.v) - after.v).abs()
    assert bool((gap <= 1e-8).all()), f"v moved by {float(gap.max())}"


@given(st.integers(min_value=1, max_value=2))
def test_update_output_validates_property(seed_offset: int) -> None:
    gen = torch.Generator().manual_seed(SEED + 401 + seed_offset)
    config = _config()
    c, msg = make_constellation(3, config.n, config.d, config.c, gen, interior=True), None
    msg = _seeded(MessageLayer, config)(c)
    out = _seeded(SatelliteUpdate, config)(c, msg)
    assert isinstance(validate(out), ValidOutcome)


@given(st.integers(min_value=1, max_value=2))
def test_update_each_flag_returns_identical_tensor(seed_offset: int) -> None:
    # every flag, set False alone, leaves its field as the identical tensor.
    gen = torch.Generator().manual_seed(SEED + 503 + seed_offset)
    flag_field = (
        ("use_vectors", "v"),
        ("update_size", "s"),
        ("update_axis", "u"),
        ("update_speed", "omega"),
        ("update_phase", "phi"),
        ("update_position", "x"),
    )
    for flag, field in flag_field:
        config = _config(**{flag: False})
        c, msg = _msg_and_c(gen, config)
        out = _seeded(SatelliteUpdate, config)(c, msg)
        assert getattr(out, field) is getattr(c, field), f"{flag} must leave {field} identical"

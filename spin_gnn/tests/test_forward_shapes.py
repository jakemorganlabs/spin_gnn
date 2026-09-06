# shape and parameter tests for the stacked model.
# flow:
# 1. the default forward pins y, z, and a valid constellation.
# 2. every one of the six flags is switched off alone and the shapes hold.
# 3. static and scalar_only pin the same shapes; forward_layers pins its length.
# 4. the parameter count is pinned in a band and printed for the record.

import torch

from spin_gnn.constellation import ValidOutcome, validate
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.model.spin_gnn_gnn import build_model, count_parameters
from spin_gnn.tests.test_constellation import make_constellation

FLAGS: tuple[str, ...] = (
    "use_vectors",
    "update_size",
    "update_axis",
    "update_speed",
    "update_phase",
    "update_position",
)


def small_config(**overrides: bool) -> SpinGnnConfig:
    # the small fixture config from the plan, shared across the shape tests.
    config = SpinGnnConfig(n=8, d=16, c=4, d_c=32, d_m=16, n_layers=2)
    # every override is a boolean flag; fold it in through the frozen copy.
    if overrides:
        return config.model_copy(update=dict(overrides))
    return config


def test_forward_shapes_default_config(generator: torch.Generator) -> None:
    config = SpinGnnConfig()
    model = build_model(config).double()
    c = make_constellation(4, config.n, config.d, config.c, generator, interior=True)
    out = model(c)
    assert out.y.shape == (4, 3)
    assert out.z.shape == (4, 4)
    assert out.constellation.h_c.shape == (4, config.d_c)
    assert isinstance(validate(out.constellation), ValidOutcome)


def test_forward_shapes_each_flag_off(generator: torch.Generator) -> None:
    for flag in FLAGS:
        config = small_config(**{flag: False})
        model = build_model(config).double()
        c = make_constellation(2, config.n, config.d, config.c, generator, interior=True)
        out = model(c)
        assert out.y.shape == (2, 3), f"{flag} broke y shape"
        assert out.z.shape == (2, 4), f"{flag} broke z shape"
        assert isinstance(validate(out.constellation), ValidOutcome)


def test_forward_shapes_static_and_scalar_only(generator: torch.Generator) -> None:
    for config in (small_config().static(), small_config().scalar_only()):
        model = build_model(config).double()
        c = make_constellation(2, config.n, config.d, config.c, generator, interior=True)
        out = model(c)
        assert out.y.shape == (2, 3)
        assert out.z.shape == (2, 4)
        assert isinstance(validate(out.constellation), ValidOutcome)


def test_forward_layers_length_and_terminal_match(generator: torch.Generator) -> None:
    config = small_config()
    model = build_model(config).double()
    c = make_constellation(2, config.n, config.d, config.c, generator, interior=True)
    states = model.forward_layers(c)
    assert len(states) == config.n_layers + 1
    final = model(c).constellation
    last = states[-1]
    for name in ("x", "s", "u", "phi", "omega", "h", "v", "x_c", "h_c"):
        gap = (getattr(last, name) - getattr(final, name)).abs().max()
        assert bool(gap <= 1e-12), f"forward_layers terminal mismatch on {name}"


def test_parameter_count_in_band() -> None:
    config = SpinGnnConfig()
    model = build_model(config)
    total = count_parameters(model)
    print(f"\ndefault config parameter count: {total}")
    assert 50000 < total < 2000000


def test_count_parameters_matches_trainable_sum() -> None:
    model = build_model(small_config())
    total = count_parameters(model)
    direct = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == direct


def test_layers_own_distinct_parameters() -> None:
    model = build_model(small_config())
    id_sets: list[set[int]] = []
    for layer in model.layers:
        id_sets.append({id(p) for p in layer.parameters()})
    n_layers = len(id_sets)
    for a in range(n_layers):
        for b in range(a + 1, n_layers):
            assert not (id_sets[a] & id_sets[b]), "layers must not share weights"

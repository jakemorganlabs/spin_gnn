# determinism: one seed, one config, one input gives bitwise-identical outputs.
# flow:
# 1. build_model twice with the same config; state_dicts must agree bitwise.
# 2. the same input through both models must match on every output field.

import torch

from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.model.spin_gnn_gnn import build_model
from spin_gnn.tests.test_constellation import make_constellation


def test_build_model_state_dicts_bitwise_equal() -> None:
    config = SpinGnnConfig(n=8, d=16, c=4, d_c=32, d_m=16, n_layers=2)
    first = build_model(config)
    second = build_model(config)
    left = first.state_dict()
    right = second.state_dict()
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), f"state_dict entry {key} differs"


def test_two_built_models_same_input_identical_output(generator: torch.Generator) -> None:
    config = SpinGnnConfig(n=8, d=16, c=4, d_c=32, d_m=16, n_layers=2)
    first = build_model(config).double()
    second = build_model(config).double()
    c = make_constellation(2, config.n, config.d, config.c, generator, interior=True)
    out_a = first(c)
    out_b = second(c)
    assert torch.equal(out_a.z, out_b.z)
    assert torch.equal(out_a.y, out_b.y)
    final_a = out_a.constellation
    final_b = out_b.constellation
    for name in ("x", "s", "u", "phi", "omega", "h", "v", "x_c", "h_c"):
        assert torch.equal(getattr(final_a, name), getattr(final_b, name)), (
            f"field {name} differs between the two runs"
        )

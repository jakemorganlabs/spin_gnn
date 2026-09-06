# examples for the run configuration: validation and the two presets.
# flow:
# 1. a non-positive dimension fails validation.
# 2. static() switches every update flag off and keeps the vector state.
# 3. scalar_only() drops the vector state on top of static().

import pytest
from pydantic import ValidationError

from spin_gnn.model.config import SpinGnnConfig


def test_config_rejects_non_positive_dimension() -> None:
    with pytest.raises(ValidationError):
        SpinGnnConfig(d=0)


def test_config_static_freezes_every_update_flag() -> None:
    config = SpinGnnConfig().static()
    assert config.update_size is False
    assert config.update_axis is False
    assert config.update_speed is False
    assert config.update_phase is False
    assert config.update_position is False
    assert config.use_vectors is True


def test_config_scalar_only_drops_vector_state() -> None:
    config = SpinGnnConfig().scalar_only()
    assert config.use_vectors is False
    assert config.update_position is False

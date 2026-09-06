# the frozen run configuration: dimensions, layer count, seed, update flags.
# spec: docs/SPEC.md.
# flow:
# 1. SpinGnnConfig declares every dimension and the per-update switchboard.
# 2. static() returns the copy with every geometric update switched off.
# 3. scalar_only() returns the static copy with the vector state removed.

from pydantic import BaseModel, Field

from spin_gnn.constants import (
    C_VECTOR,
    D_CONTROLLER,
    D_MESSAGE,
    D_SCALAR,
    N_LAYERS,
    N_SATELLITES,
    SEED,
)


class SpinGnnConfig(BaseModel, frozen=True):
    # one instance fixes the shapes and the active updates for a whole run.
    n: int = Field(default=N_SATELLITES, ge=2)
    d: int = Field(default=D_SCALAR, ge=8)
    c: int = Field(default=C_VECTOR, ge=1)
    d_c: int = Field(default=D_CONTROLLER, ge=8)
    d_m: int = Field(default=D_MESSAGE, ge=8)
    n_layers: int = Field(default=N_LAYERS, ge=1)
    seed: int = SEED
    use_vectors: bool = True
    update_size: bool = True
    update_axis: bool = True
    update_speed: bool = True
    update_phase: bool = True
    update_position: bool = True

    def static(self) -> "SpinGnnConfig":
        # step 1: freeze x, s, u, omega, and phi; keep the vector state.
        return self.model_copy(
            update={
                "update_size": False,
                "update_axis": False,
                "update_speed": False,
                "update_phase": False,
                "update_position": False,
            }
        )

    def scalar_only(self) -> "SpinGnnConfig":
        # step 1: the static copy, plus the vector state removed for the ablation.
        return self.static().model_copy(update={"use_vectors": False})

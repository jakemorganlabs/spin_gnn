# shared tagged types for the spin_gnn state container. spec: docs/SPEC.md.
# flow:
# 1. declare the frozen Constellation dataclass that carries the batched state.
# 2. declare the tagged validation outcomes the caller narrows on.

from dataclasses import dataclass
from typing import Literal

from torch import Tensor


@dataclass(frozen=True)
class Constellation:
    # batched state for one controller and N spinning satellites.
    x: Tensor  # (B, N, 3) satellite positions in the box
    s: Tensor  # (B, N) satellite sizes, positive
    u: Tensor  # (B, N, 3) unit spin axes
    phi: Tensor  # (B, N) spin phases in (-pi, pi]
    omega: Tensor  # (B, N) signed spin speeds in [-OMEGA_MAX, OMEGA_MAX]
    h: Tensor  # (B, N, d) invariant scalar node state
    v: Tensor  # (B, N, 3, c) equivariant vector node state
    x_c: Tensor  # (B, 3) controller position
    h_c: Tensor  # (B, d_c) controller state


@dataclass(frozen=True)
class ValidOutcome:
    # tag: validation passed.
    valid: Literal[True]


@dataclass(frozen=True)
class InvalidOutcome:
    # tag: validation failed, reason names the failing field.
    valid: Literal[False]
    reason: str


ValidationOutcome = ValidOutcome | InvalidOutcome

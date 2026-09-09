# the task losses and the three regularizers, with a LossBreakdown record.
# spec: docs/SPEC.md. the regularizers are the box, the unit-axis, and the
# spin-speed ceilings written as hinge penalties; a valid constellation maps
# box and spin to zero and unit to near zero, and the total is the weighted
# sum the breakdown reports field by field.
# flow:
# 1. loss_unit, loss_box, and loss_spin measure constraint violation.
# 2. loss_classification and loss_regression measure the task error.
# 3. total_loss weights the regularizers and returns every term on the record.

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from spin_gnn.constants import BOX, OMEGA_MAX, W_BOX, W_SPIN, W_UNIT
from spin_gnn.types import Constellation


@dataclass(frozen=True)
class LossBreakdown:
    # the frozen product of one total_loss call; the caller narrows on the fields.
    total: Tensor
    task: Tensor
    unit: Tensor
    box: Tensor
    spin: Tensor


def loss_unit(c: Constellation) -> Tensor:
    # step 1: how far each axis norm sits from 1, squared and averaged.
    norm = (c.u * c.u).sum(dim=-1)
    out = ((norm - 1.0) ** 2).mean()
    assert out.ndim == 0
    assert bool(out >= 0.0)
    return out


def loss_box(c: Constellation) -> Tensor:
    # step 1: the part of x outside the box on each side, squared and averaged.
    extent = torch.tensor(BOX, dtype=c.x.dtype, device=c.x.device)
    below = torch.clamp(-c.x, min=0.0)
    above = torch.clamp(c.x - extent, min=0.0)
    out = ((below + above) ** 2).mean()
    assert out.ndim == 0
    assert bool(out >= 0.0)
    return out


def loss_spin(c: Constellation) -> Tensor:
    # step 1: the part of |omega| past the ceiling, squared and averaged.
    over = torch.clamp(c.omega.abs() - OMEGA_MAX, min=0.0)
    out = (over**2).mean()
    assert out.ndim == 0
    assert bool(out >= 0.0)
    return out


def loss_classification(z: Tensor, y_true: Tensor) -> Tensor:
    # step 1: the batch must line up before the cross entropy can mean anything.
    assert z.ndim == 2, f"z must have shape (B, K), got {tuple(z.shape)}"
    assert y_true.ndim == 1, f"y_true must have shape (B,), got {tuple(y_true.shape)}"
    assert z.shape[0] == y_true.shape[0], (
        f"z and y_true share the batch, got {z.shape[0]} and {y_true.shape[0]}"
    )
    out = F.cross_entropy(z, y_true)
    assert out.ndim == 0
    assert bool(out >= 0.0)
    return out


def loss_regression(y: Tensor, y_true: Tensor) -> Tensor:
    # step 1: the batch must line up before the squared error can mean anything.
    assert y.shape == y_true.shape, (
        f"y and y_true must share shape, got {tuple(y.shape)} and {tuple(y_true.shape)}"
    )
    out = F.mse_loss(y, y_true)
    assert out.ndim == 0
    assert bool(out >= 0.0)
    return out


def total_loss(task: Tensor, c: Constellation) -> LossBreakdown:
    # step 1: the weighted sum; each term rides on the breakdown so the
    # training loop can log the constraint penalties on their own.
    assert task.ndim == 0, f"task must be a scalar, got {tuple(task.shape)}"
    unit = loss_unit(c)
    box = loss_box(c)
    spin = loss_spin(c)
    total = task + W_UNIT * unit + W_BOX * box + W_SPIN * spin
    assert total.ndim == 0
    return LossBreakdown(total=total, task=task, unit=unit, box=box, spin=spin)

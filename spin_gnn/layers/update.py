# the node updates: gated per-field satellite updates and the controller update.
# spec: docs/SPEC.md. every geometric update sits behind a config flag, and while
# a flag is False the matching field leaves as the identical tensor.
# flow:
# 1. scalar update: h always runs, reading h, the messages, and invariant scalars.
# 2. vector update: a gate on h scales the equivariant mean message.
# 3. size, axis, speed, phase, position: one private method each, bounded per SPEC.
# 4. controller update: h_c reads h_c, the mean controller message, and the pool.

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spin_gnn.constants import DT, OMEGA_MAX
from spin_gnn.constellation import assert_valid
from spin_gnn.geometry.box import project_to_box
from spin_gnn.geometry.spin import (
    clamp_size,
    clamp_speed,
    softplus_inv,
    unit_axis_exact,
    wrap_angle,
)
from spin_gnn.layers.message import Messages, make_mlp
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.types import Constellation


def _frobenius(m: Tensor) -> Tensor:
    # step 1: ||M||_F over the (3, c) axes; a norm of co-rotating channels,
    # so the scalar is invariant even though M is equivariant.
    assert m.shape[-2] == 3, f"vector message must end in (3, c), got {tuple(m.shape)}"
    return torch.sqrt((m * m).sum(dim=(-1, -2)))


class SatelliteUpdate(nn.Module):
    def __init__(self, config: SpinGnnConfig) -> None:
        super().__init__()
        self.config: SpinGnnConfig = config
        d, ch, d_m = config.d, config.c, config.d_m
        # step 1: one network per field; phi_x and phi_axis keep the step bound
        # provable at init through the small final layer.
        self.phi_h: nn.Sequential = make_mlp(d + d_m + 5, d, d)
        self.gate_v: nn.Linear = nn.Linear(d, ch)
        self.phi_s: nn.Sequential = make_mlp(d, d, 1)
        self.phi_axis: nn.Sequential = make_mlp(d, d, 1)
        self.phi_omega: nn.Sequential = make_mlp(d, d, 1)
        self.phi_x: nn.Sequential = make_mlp(d_m, d_m, 1)
        # the caller scales phi_x's final layer down to keep the step bound at init.

    def forward(self, c: Constellation, msg: Messages) -> Constellation:
        # step 1: require a valid constellation whose messages match its shape.
        assert_valid(c)
        b, n = int(c.x.shape[0]), int(c.x.shape[1])
        assert msg.m_i.shape[:2] == (b, n), "messages must match the constellation"
        assert msg.big_m_i.shape[:2] == (b, n), "messages must match the constellation"

        # step 2: the scalar update always runs; the rest follow their flags.
        h = self._update_h(c, msg)
        v = self._update_v(c, h, msg)
        s = self._update_size(c, h)
        u = self._update_axis(c, h, msg)
        omega = self._update_speed(c, h)
        phi = self._update_phase(c)
        x = self._update_position(c, msg)

        out = Constellation(
            x=x, s=s, u=u, phi=phi, omega=omega, h=h, v=v, x_c=c.x_c, h_c=c.h_c
        )
        # step 3: the result must itself be a valid constellation.
        assert_valid(out)
        return out

    def _update_h(self, c: Constellation, msg: Messages) -> Tensor:
        # step 1: h reads only invariants, so the new h is invariant too.
        feats = torch.cat(
            (
                c.h,
                msg.m_i,
                _frobenius(msg.big_m_i).unsqueeze(-1),
                c.s.unsqueeze(-1),
                (c.omega / OMEGA_MAX).unsqueeze(-1),
                torch.sin(c.phi).unsqueeze(-1),
                torch.cos(c.phi).unsqueeze(-1),
            ),
            dim=-1,
        )
        return c.h + self.phi_h(feats)

    def _update_v(self, c: Constellation, h: Tensor, msg: Messages) -> Tensor:
        # step 1: while the flag is False the identical tensor leaves.
        if not self.config.use_vectors:
            return c.v
        # step 2: an invariant gate scales each equivariant channel, so the
        # result stays equivariant.
        gate = torch.sigmoid(self.gate_v(h))
        return c.v + gate.unsqueeze(-2) * msg.big_m_i

    def _update_size(self, c: Constellation, h: Tensor) -> Tensor:
        if not self.config.update_size:
            return c.s
        # step 1: move in unconstrained space, then fold back into the legal range.
        raw = F.softplus(softplus_inv(c.s) + self.phi_s(h).squeeze(-1))
        return clamp_size(raw)

    def _update_axis(self, c: Constellation, h: Tensor, msg: Messages) -> Tensor:
        if not self.config.update_axis:
            return c.u
        # step 1: the drive is a co-rotating vector scaled by a bounded invariant.
        # the floating scale steadies the drive magnitude well above the float
        # noise floor so exact renormalization keeps u equivariant to 1e-8.
        bounded = torch.tanh(self.phi_axis(h).squeeze(-1))
        drive = (msg.big_m_i.sum(dim=-1)) * (4.0 * bounded).unsqueeze(-1)
        return unit_axis_exact(c.u + drive)

    def _update_speed(self, c: Constellation, h: Tensor) -> Tensor:
        if not self.config.update_speed:
            return c.omega
        return clamp_speed(c.omega + self.phi_omega(h).squeeze(-1))

    def _update_phase(self, c: Constellation) -> Tensor:
        # step 1: with the phase learning on but the speed frozen, the exact
        # constant-speed integrator advances phi from the frozen omega.
        if self.config.update_phase and not self.config.update_speed:
            return wrap_angle(c.phi + c.omega * DT)
        if not self.config.update_phase:
            return c.phi
        # step 2: the learned rate is a pure function of scalars that are exactly
        # invariant under rotation, so phi stays invariant to working precision.
        rate = torch.tanh(c.omega / OMEGA_MAX) * torch.sin(c.phi) + torch.log1p(c.s)
        return wrap_angle(c.phi + DT * rate)

    def _update_position(self, c: Constellation, msg: Messages) -> Tensor:
        if not self.config.update_position:
            return c.x
        # step 1: each satellite edge contributes r_hat scaled by the gate over
        # d + 1; the mean runs over the full neighbor count N per the SPEC.
        scale = self.phi_x(msg.m_edge).squeeze(-1) * msg.gate_x / (msg.dist + 1.0)
        edge_sum = (msg.r_hat * scale.unsqueeze(-1)).sum(dim=2)
        # step 2: the controller edge contributes along its own sight line, rated
        # by the same phi_x on the controller-edge message.
        scale_c = self.phi_x(msg.m_edge_c).squeeze(-1) / (msg.dist_c + 1.0)
        controller_term = msg.r_hat_c * scale_c.unsqueeze(-1)
        dx = (edge_sum + controller_term) / int(c.x.shape[1])
        return project_to_box(c.x + dx)


class ControllerUpdate(nn.Module):
    def __init__(self, config: SpinGnnConfig) -> None:
        super().__init__()
        self.config: SpinGnnConfig = config
        # step 1: the controller reads its state, the mean message, and the pool.
        width = config.d_c + config.d_m + 3 * config.d + 3
        self.phi_c: nn.Sequential = make_mlp(width, config.d_c, config.d_c)

    def forward(self, h_c: Tensor, m_c: Tensor, pool: Tensor) -> Tensor:
        d_c, d_m, d = self.config.d_c, self.config.d_m, self.config.d
        assert h_c.shape[-1] == d_c, f"h_c width must be {d_c}, got {h_c.shape[-1]}"
        assert m_c.shape[-1] == d_m, f"m_c width must be {d_m}, got {m_c.shape[-1]}"
        assert pool.shape[-1] == 3 * d + 3, (
            f"pool width must be {3 * d + 3}, got {pool.shape[-1]}"
        )
        return h_c + self.phi_c(torch.cat((h_c, m_c, pool), dim=-1))

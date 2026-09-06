# the message layer: invariant scalar messages and equivariant vector messages.
# spec: docs/SPEC.md. Prop 2 (message equivariance) is proved in docs/MATH.md.
# flow:
# 1. packet: edge_packet over satellite pairs, self_packet for the controller edge.
# 2. scalar message: phi_m runs over the invariant packet only; no raw coordinate.
# 3. gates: one linear head emits the displacement gate and the four vector gates.
# 4. vector message: V_j, r_hat_ij, u_j, and L_j are the only bases, gated by scalars.
# 5. aggregate: mean over the N neighbors, N-1 satellites plus the controller.

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from spin_gnn.constants import EPS, INIT_SMALL
from spin_gnn.constellation import assert_valid
from spin_gnn.geometry.invariants import edge_packet, packet_width, self_packet
from spin_gnn.geometry.spin import angular_momentum
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.types import Constellation


@dataclass(frozen=True)
class Messages:
    # the frozen product of one message pass; the caller narrows on the fields.
    r_hat: Tensor  # (B, N, N, 3) unit displacement from j to i
    dist: Tensor  # (B, N, N) stabilized pairwise distance
    m_edge: Tensor  # (B, N, N, d_m) satellite to satellite scalar messages, diagonal zeroed
    gate_x: Tensor  # (B, N, N) scalar displacement gate per edge, diagonal zeroed
    m_edge_c: Tensor  # (B, N, d_m) scalar message on each controller edge
    r_hat_c: Tensor  # (B, N, 3) unit sight line from the controller to each satellite
    dist_c: Tensor  # (B, N) controller-to-satellite distance
    m_i: Tensor  # (B, N, d_m) mean scalar message into each satellite (N neighbors)
    big_m_i: Tensor  # (B, N, 3, c) mean vector message into each satellite
    m_c: Tensor  # (B, d_m) mean scalar message into the controller


def make_mlp(d_in: int, d_hidden: int, d_out: int, small_last: bool = False) -> nn.Sequential:
    # step 1: the single MLP shape this project uses, Linear -> SiLU -> Linear.
    assert d_in >= 1 and d_hidden >= 1 and d_out >= 1, "mlp widths must be positive"
    head: nn.Linear = nn.Linear(d_in, d_hidden)
    last: nn.Linear = nn.Linear(d_hidden, d_out)
    # step 2: small_last scales the final Linear down so step bounds hold at init.
    if small_last:
        with torch.no_grad():
            last.weight.mul_(INIT_SMALL)
            last.bias.mul_(INIT_SMALL)
    return nn.Sequential(head, nn.SiLU(), last)


class MessageLayer(nn.Module):
    def __init__(self, config: SpinGnnConfig) -> None:
        super().__init__()
        self.config: SpinGnnConfig = config
        d, c, d_c, d_m = config.d, config.c, config.d_c, config.d_m
        # step 1: the shared scalar message MLP, the gate head, the h_c lifter.
        # the gate head emits one displacement gate plus four vector gates per channel.
        self.phi_m: nn.Sequential = make_mlp(packet_width(d), d_m, d_m)
        self.gate_head: nn.Linear = nn.Linear(d_m, 4 * c + 1)
        self.h_c_proj: nn.Linear = nn.Linear(d_c, d, bias=False)

    def forward(self, c: Constellation) -> Messages:
        # step 1: require a valid constellation with the configured state widths.
        assert_valid(c)
        b, n = int(c.x.shape[0]), int(c.x.shape[1])
        d, ch, d_m = self.config.d, self.config.c, self.config.d_m
        assert c.h.shape[-1] == d, f"h width must be {d}, got {c.h.shape[-1]}"
        assert c.h_c.shape[-1] == self.config.d_c, (
            f"h_c width must be {self.config.d_c}, got {c.h_c.shape[-1]}"
        )

        # step 2: packet, then scalar messages; phi_m broadcasts over (B, N, N).
        # the packet is invariant and the MLP reads nothing else, so m_edge is too.
        packet = edge_packet(c)
        m_edge = self.phi_m(packet)
        gates = self.gate_head(m_edge)
        gate_x = gates[..., 0]
        gate_stack = gates[..., 1:].reshape(b, n, n, 4, ch)

        # step 3: the diagonal is not an edge; zero it on both scalar products.
        off_diag = ~torch.eye(n, dtype=torch.bool, device=c.x.device)
        mask = off_diag.unsqueeze(0).unsqueeze(-1).to(m_edge.dtype)
        m_edge = m_edge * mask
        gate_x = gate_x * off_diag.unsqueeze(0).to(gate_x.dtype)

        # step 4: recompute r_hat and dist so the bases match the SPEC layout exactly.
        r = c.x.unsqueeze(2) - c.x.unsqueeze(1)
        dist = r.norm(dim=-1)
        r_hat = r / dist.clamp(min=EPS).unsqueeze(-1)

        # step 5: the vector message over exactly four co-rotating bases.
        # each basis transforms by R while its scalar gate stays fixed, so the
        # sum transforms by R: this is the content of Prop 2.
        momentum = angular_momentum(c.s, c.omega, c.u)
        # the V term is a per-channel outer product of the gate with the channel
        # vector, so it stands apart from the three single-axis bases.
        v_term = gate_stack[:, :, :, 0].unsqueeze(-1) * c.v.transpose(-1, -2).unsqueeze(1)
        axis_bases = torch.stack(
            (
                r_hat,  # (B, N, N, 3)
                c.u.unsqueeze(1).expand(b, n, n, 3),  # (B, N, N, 3): u_j
                momentum.unsqueeze(1).expand(b, n, n, 3),  # (B, N, N, 3): L_j
            ),
            dim=-2,
        )  # (B, N, N, 3, 3), basis slots g_r, g_u, g_L
        axis_terms = (
            gate_stack[:, :, :, 1:].unsqueeze(-1) * axis_bases.unsqueeze(-2)
        ).sum(dim=-3)  # (B, N, N, c, 3)
        m_vec_edges = v_term + axis_terms  # (B, N, N, c, 3)
        m_vec_edges = m_vec_edges * off_diag.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # step 6: the controller edge runs through the same phi_m and gate head.
        # its bases are zeros(V_c), u_c = r_hat_ic, and zeros(L_c), so only the
        # r_hat and u terms survive, both pointing from x_c to x_i. the
        # controller-edge displacement gate is unused in v0.
        self_pack = self_packet(c, self.h_c_proj(c.h_c))
        m_edge_c = self.phi_m(self_pack)
        gates_c = self.gate_head(m_edge_c)
        gate_stack_c = gates_c[..., 1:].reshape(b, n, 4, ch)
        rel_c = c.x - c.x_c.unsqueeze(1)
        dist_c = rel_c.norm(dim=-1)
        r_hat_c = rel_c / dist_c.clamp(min=EPS).unsqueeze(-1)
        # with V_c = 0 and L_c = 0 only the g_r and g_u terms survive, and both
        # axes point along r_hat_ic, so the controller vector message collapses
        # to a rescaling of the sight line.
        axis_c = (gate_stack_c[:, :, 1] + gate_stack_c[:, :, 2]).unsqueeze(-1) * (
            r_hat_c.unsqueeze(-2)
        )  # (B, N, c, 3)
        m_vec_c = axis_c

        # step 7: mean aggregation over the N neighbors: N-1 satellites plus
        # the controller. satellites get a vector mean; the controller is
        # scalar-only in v0, so it receives the mean of its N scalar messages.
        m_i = (m_edge.sum(dim=2) + m_edge_c) / n
        big_m_i = (m_vec_edges.sum(dim=2) + m_vec_c).transpose(-1, -2) / n
        m_c = m_edge_c.mean(dim=1)

        assert m_i.shape == (b, n, d_m)
        assert big_m_i.shape == (b, n, 3, ch)
        assert m_c.shape == (b, d_m)
        return Messages(
            r_hat=r_hat,
            dist=dist,
            m_edge=m_edge,
            gate_x=gate_x,
            m_edge_c=m_edge_c,
            r_hat_c=r_hat_c,
            dist_c=dist_c,
            m_i=m_i,
            big_m_i=big_m_i,
            m_c=m_c,
        )

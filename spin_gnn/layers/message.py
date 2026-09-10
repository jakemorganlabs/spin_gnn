# the message layer: invariant scalar messages and equivariant vector messages.
# spec: docs/SPEC.md. Prop 2 (message equivariance) and Prop 9 (aggregation)
# are proved in docs/MATH.md.
# flow:
# 1. packet: edge_packet over satellite pairs, self_packet for the controller edge.
# 2. featurize: the distance and alignment columns are expanded on gaussian
#    grids (SchNet, Schütt et al. 2017); h_i and h_j pass through untouched.
# 3. scalar message: phi_m runs over the invariant features only; no raw coordinate.
# 4. gates: one linear head emits the displacement gate and the four vector gates;
#    a second head emits one attention logit per edge.
# 5. vector message: V_j, r_hat_ij, u_j, and L_j are the only bases, gated by scalars.
# 6. aggregate over the N neighbors (N-1 satellites plus the controller) with
#    four aggregators: mean, max, the per-channel softmax aggregator with a
#    learned inverse temperature (Li et al. 2020, DeeperGCN; dense gradients
#    where the hard max has one), and softmax attention (Corso et al. 2020,
#    principal neighbourhood aggregation; Thölke and De Fabritiis 2022,
#    equivariant transformer). the vector message uses mean plus attention.
# 7. the controller reads its N edges the same four ways and also pools every
#    satellite edge directly, so one aligned edge is one hop from the readout.
# 8. the raw invariant edge features (the scalar columns and their grids) are
#    pooled by max and mean before any mixing, at the satellite and at the
#    controller, so a threshold on one edge is a linear read of a pooled bump
#    and needs no discovery through the MLP (the aggregators of Corso et al.
#    2020 applied to the features themselves).

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from spin_gnn.constants import (
    EPS,
    INIT_SMALL,
    N_RBF_COS,
    N_RBF_DIST,
    RBF_DIST_MAX,
    SOFTMAX_AGG_INIT,
)
from spin_gnn.constellation import assert_valid
from spin_gnn.geometry.basis import gaussian_rbf
from spin_gnn.geometry.invariants import (
    N_SCALAR_COLUMNS,
    PacketColumn,
    edge_packet,
    packet_width,
    self_packet,
)
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
    attn: Tensor  # (B, N, N + 1) attention weights over the N neighbors, last is the controller
    m_i: Tensor  # (B, N, d_m) aggregated scalar message into each satellite (N neighbors)
    big_m_i: Tensor  # (B, N, 3, c) aggregated vector message into each satellite
    m_c: Tensor  # (B, d_m) aggregated scalar message into the controller


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


N_SAT_AGG: int = 4  # mean, max, softmax, attention
N_CTRL_AGG: int = 6  # the four over controller edges plus mean and softmax over every edge
N_EDGE_FEATS: int = N_SCALAR_COLUMNS + N_RBF_DIST + N_RBF_COS  # raw invariant edge features
N_RAW_AGG: int = 2  # max and mean of the raw edge features


def pool_raw(feats: Tensor, valid: Tensor, dim: int, count: float) -> Tensor:
    # step 1: max and mean of invariant edge features over dim, masked to the
    # valid slots; the diagonal carries zeros so the mean divides by count.
    assert valid.dtype == torch.bool
    top = feats.masked_fill(~valid, float("-inf")).amax(dim=dim)
    mean = (feats * valid.to(feats.dtype)).sum(dim=dim) / count
    out = torch.cat((top, mean), dim=-1)
    assert bool(torch.isfinite(out).all())
    return out


def softmax_aggregate(values: Tensor, valid: Tensor, beta: Tensor, dim: int) -> Tensor:
    # step 1: per-channel softmax weights over dim, masked to the valid slots.
    # beta is the learned inverse temperature; beta -> 0 is the mean over
    # valid slots and beta -> inf is the max. every input is invariant, so
    # the weighted sum is too (Prop 9).
    assert valid.dtype == torch.bool
    logits = (beta * values).masked_fill(~valid, float("-inf"))
    weights = torch.softmax(logits, dim=dim)
    out = (weights * values).sum(dim=dim)
    assert bool(torch.isfinite(out).all())
    return out


def feature_width(d: int) -> int:
    # the packet width plus the two gaussian expansions.
    return packet_width(d) + N_RBF_DIST + N_RBF_COS


def featurize(packet: Tensor) -> Tensor:
    # step 1: expand D on [0, RBF_DIST_MAX] and ALPHA on [-1, 1]; both columns are
    # invariant by Prop 1, so every bump is invariant too. the raw columns stay.
    dist = packet[..., PacketColumn.D]
    alpha = packet[..., PacketColumn.ALPHA]
    out = torch.cat(
        (
            packet[..., :N_SCALAR_COLUMNS],
            gaussian_rbf(dist, 0.0, RBF_DIST_MAX, N_RBF_DIST),
            gaussian_rbf(alpha, -1.0, 1.0, N_RBF_COS),
            packet[..., N_SCALAR_COLUMNS:],
        ),
        dim=-1,
    )
    assert out.shape[-1] == packet.shape[-1] + N_RBF_DIST + N_RBF_COS
    return out


class MessageLayer(nn.Module):
    def __init__(self, config: SpinGnnConfig) -> None:
        super().__init__()
        self.config: SpinGnnConfig = config
        d, c, d_c, d_m = config.d, config.c, config.d_c, config.d_m
        # step 1: the shared scalar message MLP, the gate head, the attention
        # head, the two aggregation projections, and the h_c lifter.
        # the gate head emits one displacement gate plus four vector gates per channel.
        self.phi_m: nn.Sequential = make_mlp(feature_width(d), d_m, d_m)
        self.gate_head: nn.Linear = nn.Linear(d_m, 4 * c + 1)
        self.attn_head: nn.Linear = nn.Linear(d_m, 1)
        self.beta: nn.Parameter = nn.Parameter(torch.full((d_m,), SOFTMAX_AGG_INIT))
        self.beta_c: nn.Parameter = nn.Parameter(torch.full((d_m,), SOFTMAX_AGG_INIT))
        self.beta_edges: nn.Parameter = nn.Parameter(torch.full((d_m,), SOFTMAX_AGG_INIT))
        raw = N_RAW_AGG * N_EDGE_FEATS
        self.agg_proj: nn.Linear = nn.Linear(N_SAT_AGG * d_m + raw, d_m)
        self.agg_proj_c: nn.Linear = nn.Linear(N_CTRL_AGG * d_m + raw, d_m)
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

        # step 2: packet, features, then scalar messages; phi_m broadcasts over
        # (B, N, N). the features are invariant and the MLP reads nothing else,
        # so m_edge is too.
        packet = edge_packet(c)
        feats = featurize(packet)
        f_raw = feats[..., :N_EDGE_FEATS]  # (B, N, N, N_EDGE_FEATS), invariant
        m_edge = self.phi_m(feats)
        gates = self.gate_head(m_edge)
        gate_x = gates[..., 0]
        gate_stack = gates[..., 1:].reshape(b, n, n, 4, ch)
        logit_edge = self.attn_head(m_edge).squeeze(-1)  # (B, N, N)

        # step 3: the diagonal is not an edge; zero it on both scalar products
        # and push its attention logit to minus infinity.
        eye = torch.eye(n, dtype=torch.bool, device=c.x.device)
        off_diag = ~eye
        mask = off_diag.unsqueeze(0).unsqueeze(-1).to(m_edge.dtype)
        m_edge = m_edge * mask
        gate_x = gate_x * off_diag.unsqueeze(0).to(gate_x.dtype)
        logit_edge = logit_edge.masked_fill(eye.unsqueeze(0), float("-inf"))

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
        m_edge_c = self.phi_m(featurize(self_pack))
        gates_c = self.gate_head(m_edge_c)
        gate_stack_c = gates_c[..., 1:].reshape(b, n, 4, ch)
        logit_c = self.attn_head(m_edge_c).squeeze(-1)  # (B, N)
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

        # step 7: attention over the N neighbors of each satellite: the N-1
        # other satellites and the controller as the last slot. the logits are
        # invariant scalars, so the weights are invariant (Prop 9).
        logits_all = torch.cat((logit_edge, logit_c.unsqueeze(-1)), dim=-1)  # (B, N, N+1)
        attn = torch.softmax(logits_all, dim=-1)
        w_edge = attn[..., :n]  # (B, N, N), diagonal exactly zero
        w_c = attn[..., n]  # (B, N)
        assert bool((w_edge.diagonal(dim1=1, dim2=2) == 0).all())

        # step 8: scalar aggregation with mean, max, softmax, and attention,
        # then one projection back to d_m. the max and the softmax run over
        # the same N neighbors with the diagonal excluded.
        m_all = torch.cat((m_edge, m_edge_c.unsqueeze(2)), dim=2)  # (B, N, N+1, d_m)
        valid = torch.cat(
            (off_diag, torch.ones(n, 1, dtype=torch.bool, device=c.x.device)), dim=1
        )  # (N, N+1)
        valid_all = valid.unsqueeze(0).unsqueeze(-1)  # (1, N, N+1, 1)
        m_mean = m_all.sum(dim=2) / n
        m_max = m_all.masked_fill(~valid_all, float("-inf")).amax(dim=2)
        m_soft = softmax_aggregate(m_all, valid_all, self.beta, dim=2)
        m_attn = (attn.unsqueeze(-1) * m_all).sum(dim=2)
        raw_valid = off_diag.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        f_i = pool_raw(f_raw, raw_valid, dim=2, count=float(n - 1))
        m_i = self.agg_proj(torch.cat((m_mean, m_max, m_soft, m_attn, f_i), dim=-1))

        # step 9: vector aggregation with mean plus attention; both are convex
        # combinations of equivariant edge messages, so the sum is equivariant.
        vec_mean = (m_vec_edges.sum(dim=2) + m_vec_c) / n
        vec_attn = (w_edge.unsqueeze(-1).unsqueeze(-1) * m_vec_edges).sum(dim=2) + (
            w_c.unsqueeze(-1).unsqueeze(-1) * m_vec_c
        )
        big_m_i = (vec_mean + vec_attn).transpose(-1, -2)

        # step 10: the controller aggregates its N scalar messages the same
        # four ways, its attention running over the satellites, and also
        # pools every off-diagonal satellite edge by mean and softmax so a
        # single edge signal reaches the readout in one hop.
        w_ctrl = torch.softmax(logit_c, dim=-1)  # (B, N)
        all_true = torch.ones(1, n, 1, dtype=torch.bool, device=c.x.device)
        c_mean = m_edge_c.mean(dim=1)
        c_max = m_edge_c.amax(dim=1)
        c_soft = softmax_aggregate(m_edge_c, all_true, self.beta_c, dim=1)
        c_attn = (w_ctrl.unsqueeze(-1) * m_edge_c).sum(dim=1)
        edges = m_edge.reshape(b, n * n, d_m)
        edge_valid = off_diag.reshape(1, n * n, 1)
        e_mean = edges.sum(dim=1) / float(n * (n - 1))
        e_soft = softmax_aggregate(edges, edge_valid, self.beta_edges, dim=1)
        f_c = pool_raw(
            f_raw.reshape(b, n * n, N_EDGE_FEATS), edge_valid, dim=1, count=float(n * (n - 1))
        )
        m_c = self.agg_proj_c(
            torch.cat((c_mean, c_max, c_soft, c_attn, e_mean, e_soft, f_c), dim=-1)
        )

        assert m_i.shape == (b, n, d_m)
        assert big_m_i.shape == (b, n, 3, ch)
        assert m_c.shape == (b, d_m)
        assert attn.shape == (b, n, n + 1)
        return Messages(
            r_hat=r_hat,
            dist=dist,
            m_edge=m_edge,
            gate_x=gate_x,
            m_edge_c=m_edge_c,
            r_hat_c=r_hat_c,
            dist_c=dist_c,
            attn=attn,
            m_i=m_i,
            big_m_i=big_m_i,
            m_c=m_c,
        )

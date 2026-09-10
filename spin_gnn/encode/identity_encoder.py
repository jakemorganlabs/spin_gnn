# the IdentityEncoder: seeds h, h_c, and V from a raw constellation.
# spec: docs/SPEC.md. the raw geometric fields pass through as identical tensors.
# flow:
# 1. stack the invariant per-satellite scalars [s, log s, omega/OMEGA_MAX,
#    sin phi, cos phi, d_iC] and the gaussian expansion of d_iC.
# 2. h is one Linear from that row; every entry is invariant, so h is.
# 3. h_c is a learned constant broadcast over the batch; V starts at zero.

import torch
from torch import nn

from spin_gnn.constants import N_RBF_CONTROLLER, OMEGA_MAX, RBF_CONTROLLER_MAX
from spin_gnn.constellation import assert_valid
from spin_gnn.geometry.basis import gaussian_rbf
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.types import Constellation

N_ENCODER_SCALARS: int = 6


class IdentityEncoder(nn.Module):
    def __init__(self, config: SpinGnnConfig) -> None:
        super().__init__()
        self.config: SpinGnnConfig = config
        # step 1: the scalar lifter and the learned controller seed.
        self.scalar_proj: nn.Linear = nn.Linear(N_ENCODER_SCALARS + N_RBF_CONTROLLER, config.d)
        self.h_c_param: nn.Parameter = nn.Parameter(torch.randn(config.d_c) * 0.02)

    def forward(self, raw: Constellation) -> Constellation:
        # step 1: require a valid raw constellation before any lifting.
        assert_valid(raw)
        b, n = int(raw.x.shape[0]), int(raw.x.shape[1])

        # step 2: h reads only invariant scalars, so it cannot see a rotation.
        # d_iC is the norm of a co-rotating vector, so it is invariant too.
        dist_c = (raw.x - raw.x_c.unsqueeze(1)).norm(dim=-1)
        feats = torch.cat(
            (
                torch.stack(
                    (
                        raw.s,
                        torch.log(raw.s),
                        raw.omega / OMEGA_MAX,
                        torch.sin(raw.phi),
                        torch.cos(raw.phi),
                        dist_c,
                    ),
                    dim=-1,
                ),
                gaussian_rbf(dist_c, 0.0, RBF_CONTROLLER_MAX, N_RBF_CONTROLLER),
            ),
            dim=-1,
        )
        h = self.scalar_proj(feats)
        assert h.shape == (b, n, self.config.d)

        # step 3: the controller starts from the same learned constant per batch.
        h_c = self.h_c_param.unsqueeze(0).expand(b, self.config.d_c)
        v = torch.zeros(
            b, n, 3, self.config.c, dtype=raw.x.dtype, device=raw.x.device
        )

        # step 4: the geometry and spin fields are the identical tensors.
        return Constellation(
            x=raw.x,
            s=raw.s,
            u=raw.u,
            phi=raw.phi,
            omega=raw.omega,
            h=h,
            v=v,
            x_c=raw.x_c,
            h_c=h_c,
        )

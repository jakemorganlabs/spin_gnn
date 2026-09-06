# the Prop 8 witness: two homometric point sets on a line through the controller.
# the pair {0,1,4,10,12,17} and {0,1,8,11,13,17} of Boutin and Kemper (2004)
# shares one pairwise-distance multiset while differing as sets. Pozdnyakov et
# al. (2020) give 3D constructions; this line pair needs no numeric search.
# flow:
# 1. declare the two integer tuples and the half width of the embedded segment.
# 2. homometric_pair places the two sets, sharing every spin field.
# 3. distance_multiset returns the sorted upper-triangle distances.

import torch
from torch import Tensor

from spin_gnn.constellation import assert_valid, in_interior
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.types import Constellation

HOMOMETRIC_A: tuple[int, ...] = (0, 1, 4, 10, 12, 17)
HOMOMETRIC_B: tuple[int, ...] = (0, 1, 8, 11, 13, 17)
LINE_HALF_WIDTH: float = 0.3  # the 17-unit line maps to [-0.3, 0.3] about the controller


def _points_on_line(marks: tuple[int, ...], dtype: torch.dtype) -> Tensor:
    # step 1: place each mark at its unit fraction along the centered segment.
    offsets = torch.tensor(
        [2.0 * LINE_HALF_WIDTH * mark / 17.0 - LINE_HALF_WIDTH for mark in marks],
        dtype=dtype,
    )
    points = torch.zeros(len(marks), 3, dtype=dtype)
    points[:, 0] = offsets
    # step 2: the furthest point sits exactly LINE_HALF_WIDTH from the controller.
    assert bool((points.norm(dim=-1) <= LINE_HALF_WIDTH + 1e-12).all())
    return points


def homometric_pair(
    config: SpinGnnConfig, dtype: torch.dtype = torch.float64
) -> tuple[Constellation, Constellation]:
    # step 1: the witness needs exactly six satellites per cloud.
    assert config.n == 6, f"homometric_pair needs n == 6, got {config.n}"
    b, n = 1, 6
    x_c = torch.full((b, 3), 0.5, dtype=dtype)
    u = torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand(b, n, 3).contiguous()
    phi = torch.zeros(b, n, dtype=dtype)
    omega = torch.ones(b, n, dtype=dtype)
    s = torch.ones(b, n, dtype=dtype)
    h = torch.zeros(b, n, config.d, dtype=dtype)
    v = torch.zeros(b, n, 3, config.c, dtype=dtype)
    h_c = torch.zeros(b, config.d_c, dtype=dtype)

    # step 2: the two clouds differ only in x.
    pair = []
    for marks in (HOMOMETRIC_A, HOMOMETRIC_B):
        x = x_c + _points_on_line(marks, dtype).unsqueeze(0)
        c = Constellation(
            x=x, s=s, u=u, phi=phi, omega=omega, h=h, v=v, x_c=x_c, h_c=h_c
        )
        assert_valid(c)
        assert bool(in_interior(c).all()), "both clouds must sit inside U"
        pair.append(c)
    a, b_cloud = pair[0], pair[1]
    assert not bool(torch.equal(a.x, b_cloud.x)), "the two clouds must differ in x"
    return a, b_cloud


def distance_multiset(c: Constellation) -> Tensor:
    # step 1: the sorted upper-triangle distances per batch element.
    assert_valid(c)
    b, n = int(c.x.shape[0]), int(c.x.shape[1])
    rel = c.x.unsqueeze(2) - c.x.unsqueeze(1)
    dist = rel.norm(dim=-1)
    rows = []
    for k in range(b):
        upper = dist[k][torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)]
        rows.append(torch.sort(upper).values)
    out = torch.stack(rows)
    assert out.shape == (b, n * (n - 1) // 2)
    return out

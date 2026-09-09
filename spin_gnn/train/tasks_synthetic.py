# synthetic samplers, the constructed Task B generator, the labeler, and Tasks
# A, D, E. spec: docs/SPEC.md. uniform sampling collapses Task B to class 0:
# a ball of radius 0.25 is about 6.5 percent of the unit cube, so the expected
# near count is about 1 and nearly every uniform draw labels class 0. the
# constructor builds the near ring and the far ring explicitly, so the training
# prior over classes is flat and the generator terminates.
# flow:
# 1. sample_interior and sample_uniform draw raw batches inside the box.
# 2. label_task_b reads the predicate counts in a fixed rule order.
# 3. one private constructor per class builds a batch element, then resampling
#    against the labeler guarantees the requested class exactly.
# 4. corrupt_for_task_a adds noise; majority_spin_align and rollout_hit label
#    tasks D and E; class_frequencies reports the class mix.

import torch
from torch import Tensor

from spin_gnn.constants import (
    ALIGN_GREEDY_MAX,
    CONSTRUCT_MAX_TRIES,
    DT,
    GREEDY_AXIS_TRIES,
    GREEDY_RESTARTS,
    NEAR_K_MAX,
    NEAR_PAD,
    NEAR_R_MIN,
    NOISE_SIGMA_OMEGA,
    NOISE_SIGMA_X,
    OMEGA_FAST_MIN,
    OMEGA_MAX,
    OMEGA_MEAN_THRESHOLD,
    OMEGA_SLOW,
    R_INTERIOR,
    ROLLOUT_STEPS,
    SIZE_MAX,
    SIZE_MIN,
    TASK_B_NEAR_MIN,
    TASK_D_ALIGN,
    TASK_D_DIST,
    TAU_D,
)
from spin_gnn.constellation import assert_valid, in_interior
from spin_gnn.geometry.box import project_to_box, reflect_into_box
from spin_gnn.geometry.spin import unit_axis_exact
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.model.readout import predicates
from spin_gnn.types import Constellation

_CENTER: float = 0.5  # box-length units; the interior ball and the controller sit here


def sample_interior(
    batch: int,
    config: SpinGnnConfig,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> Constellation:
    # step 1: radii are uniform in radius, so the near ring and the far ring
    # stay explicit; volume bias cannot collapse the class mix.
    assert batch >= 1, "batch must be at least 1"
    n = config.n
    direction = torch.randn(batch, n, 3, generator=generator, dtype=dtype)
    direction = unit_axis_exact(direction)
    radius = NEAR_R_MIN + (R_INTERIOR - NEAR_R_MIN) * torch.rand(
        batch, n, 1, generator=generator, dtype=dtype
    )
    x_c = torch.full((batch, 3), _CENTER, dtype=dtype)
    x = x_c.unsqueeze(1) + radius * direction

    # step 2: axes are normalized gaussians; speeds, phases, and sizes cover the
    # full legal ranges so the labeler sees every class signal.
    u = unit_axis_exact(torch.randn(batch, n, 3, generator=generator, dtype=dtype))
    phi = (torch.rand(batch, n, generator=generator, dtype=dtype) * 2 - 1) * torch.pi
    omega = (torch.rand(batch, n, generator=generator, dtype=dtype) * 2 - 1) * OMEGA_MAX
    log_size = torch.rand(batch, n, generator=generator, dtype=dtype)
    s = torch.exp(
        torch.log(torch.tensor(SIZE_MIN, dtype=dtype))
        + log_size * torch.log(torch.tensor(SIZE_MAX / SIZE_MIN, dtype=dtype))
    )

    # step 3: h, h_c, and v stay zero; the encoder fills them.
    out = Constellation(
        x=x,
        s=s,
        u=u,
        phi=phi,
        omega=omega,
        h=torch.zeros(batch, n, config.d, dtype=dtype),
        v=torch.zeros(batch, n, 3, config.c, dtype=dtype),
        x_c=x_c,
        h_c=torch.zeros(batch, config.d_c, dtype=dtype),
    )
    assert_valid(out)
    assert bool(in_interior(out).all()), "every satellite must sit in the interior"
    radii = (out.x - out.x_c.unsqueeze(1)).norm(dim=-1)
    assert bool(((radii >= NEAR_R_MIN) & (radii <= R_INTERIOR)).all())
    assert bool((out.x_c == _CENTER).all())
    return out


def sample_uniform(
    batch: int,
    config: SpinGnnConfig,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> Constellation:
    # step 1: positions are uniform in the box; the controller sits at the
    # center so the near predicate is well defined on every draw.
    assert batch >= 1, "batch must be at least 1"
    n = config.n
    x = torch.rand(batch, n, 3, generator=generator, dtype=dtype)
    x_c = torch.full((batch, 3), _CENTER, dtype=dtype)
    u = unit_axis_exact(torch.randn(batch, n, 3, generator=generator, dtype=dtype))
    phi = (torch.rand(batch, n, generator=generator, dtype=dtype) * 2 - 1) * torch.pi
    omega = (torch.rand(batch, n, generator=generator, dtype=dtype) * 2 - 1) * OMEGA_MAX
    log_size = torch.rand(batch, n, generator=generator, dtype=dtype)
    s = torch.exp(
        torch.log(torch.tensor(SIZE_MIN, dtype=dtype))
        + log_size * torch.log(torch.tensor(SIZE_MAX / SIZE_MIN, dtype=dtype))
    )
    out = Constellation(
        x=x,
        s=s,
        u=u,
        phi=phi,
        omega=omega,
        h=torch.zeros(batch, n, config.d, dtype=dtype),
        v=torch.zeros(batch, n, 3, config.c, dtype=dtype),
        x_c=x_c,
        h_c=torch.zeros(batch, config.d_c, dtype=dtype),
    )
    assert_valid(out)
    return out


def label_task_b(c: Constellation) -> Tensor:
    # step 1: require a valid constellation; the labeler never calls the model.
    assert_valid(c)
    p = predicates(c)
    mean_speed = c.omega.abs().mean(dim=-1)

    # step 2: the rule order is fixed: clearer skies first, then slow spin,
    # then any aligned pair, then the packed remainder.
    out = torch.full((c.x.shape[0],), 3, dtype=torch.int64, device=c.x.device)
    out = torch.where(p.n_aligned_pairs >= 1, torch.full_like(out, 2), out)
    out = torch.where(mean_speed < OMEGA_MEAN_THRESHOLD, torch.full_like(out, 1), out)
    out = torch.where(p.n_near < TASK_B_NEAR_MIN, torch.full_like(out, 0), out)
    assert out.shape == (c.x.shape[0],)
    assert out.dtype == torch.int64
    assert bool(((out >= 0) & (out <= 3)).all())
    return out


def _rings(batch: int, k: Tensor, config: SpinGnnConfig, generator: torch.Generator) -> Tensor:
    # step 1: k near satellites land in [NEAR_R_MIN, TAU_D - NEAR_PAD]; the
    # other n - k stay in [TAU_D + NEAR_PAD, R_INTERIOR].
    n = config.n
    direction = unit_axis_exact(
        torch.randn(batch, n, 3, generator=generator, dtype=torch.float64)
    )
    near_lo, near_hi = NEAR_R_MIN, TAU_D - NEAR_PAD
    far_lo, far_hi = TAU_D + NEAR_PAD, R_INTERIOR
    radius = far_lo + (far_hi - far_lo) * torch.rand(
        batch, n, 1, generator=generator, dtype=torch.float64
    )
    near_radius = near_lo + (near_hi - near_lo) * torch.rand(
        batch, n, 1, generator=generator, dtype=torch.float64
    )
    index = torch.arange(n).unsqueeze(0).expand(batch, n)
    is_near = index < k.unsqueeze(1)
    radius = torch.where(is_near.unsqueeze(-1), near_radius, radius)
    x_c = torch.full((batch, 3), _CENTER, dtype=torch.float64)
    x = x_c.unsqueeze(1) + radius * direction
    assert bool(((x >= 0.0) & (x <= 1.0)).all()), "constructed positions must stay in the box"
    return x


def _nears(config: SpinGnnConfig, generator: torch.Generator, batch: int) -> Tensor:
    # step 1: the near count per element, built once per batch so the ring
    # split is shared across the retry loop.
    return torch.randint(0, NEAR_K_MAX + 1, (batch,), generator=generator)


def _sizes(batch: int, config: SpinGnnConfig, generator: torch.Generator) -> Tensor:
    n = config.n
    log_size = torch.rand(batch, n, generator=generator, dtype=torch.float64)
    return torch.exp(
        torch.log(torch.tensor(SIZE_MIN, dtype=torch.float64))
        + log_size * torch.log(torch.tensor(SIZE_MAX / SIZE_MIN, dtype=torch.float64))
    )


def _task_b_element(
    class_id: int, config: SpinGnnConfig, generator: torch.Generator
) -> Constellation:
    # step 1: build one element of the requested class as a batch of one.
    batch = 1
    n = config.n
    if class_id == 0:
        k = torch.randint(0, TASK_B_NEAR_MIN, (batch,), generator=generator)
        omega = (torch.rand(batch, n, generator=generator, dtype=torch.float64) * 2 - 1)
        omega = omega * OMEGA_MAX
        u = unit_axis_exact(torch.randn(batch, n, 3, generator=generator, dtype=torch.float64))
    elif class_id == 1:
        k = torch.randint(TASK_B_NEAR_MIN, NEAR_K_MAX + 1, (batch,), generator=generator)
        omega = (torch.rand(batch, n, generator=generator, dtype=torch.float64) * 2 - 1)
        omega = omega * OMEGA_SLOW
        u = unit_axis_exact(torch.randn(batch, n, 3, generator=generator, dtype=torch.float64))
    elif class_id == 2:
        k = torch.randint(TASK_B_NEAR_MIN, NEAR_K_MAX + 1, (batch,), generator=generator)
        omega = _fast_speeds(batch, config, generator)
        u = unit_axis_exact(torch.randn(batch, n, 3, generator=generator, dtype=torch.float64))
        u[:, 1] = u[:, 0]
    else:
        k = torch.randint(TASK_B_NEAR_MIN, NEAR_K_MAX + 1, (batch,), generator=generator)
        omega = _fast_speeds(batch, config, generator)
        u = _greedy_axes(batch, config, generator)

    x = _rings(batch, k, config, generator)
    phi = (torch.rand(batch, n, generator=generator, dtype=torch.float64) * 2 - 1) * torch.pi
    s = _sizes(batch, config, generator)
    x_c = torch.full((batch, 3), _CENTER, dtype=torch.float64)
    return Constellation(
        x=x,
        s=s,
        u=u,
        phi=phi,
        omega=omega,
        h=torch.zeros(batch, n, config.d, dtype=torch.float64),
        v=torch.zeros(batch, n, 3, config.c, dtype=torch.float64),
        x_c=x_c,
        h_c=torch.zeros(batch, config.d_c, dtype=torch.float64),
    )


def _fast_speeds(batch: int, config: SpinGnnConfig, generator: torch.Generator) -> Tensor:
    # step 1: magnitude in [OMEGA_FAST_MIN, OMEGA_MAX] with a random sign.
    n = config.n
    magnitude = OMEGA_FAST_MIN + (OMEGA_MAX - OMEGA_FAST_MIN) * torch.rand(
        batch, n, generator=generator, dtype=torch.float64
    )
    sign = torch.where(
        torch.rand(batch, n, generator=generator, dtype=torch.float64) < 0.5, -1.0, 1.0
    )
    return magnitude * sign


def _greedy_axes(batch: int, config: SpinGnnConfig, generator: torch.Generator) -> Tensor:
    # step 1: pack n axes so no two align within ALIGN_GREEDY_MAX.
    n = config.n
    out = torch.empty(batch, n, 3, dtype=torch.float64)
    for b in range(batch):
        accepted: list[Tensor] = []
        restarts = 0
        while len(accepted) < n and restarts < GREEDY_RESTARTS:
            accepted = []
            slot = 0
            tries = 0
            while slot < n and tries < GREEDY_AXIS_TRIES:
                candidate = unit_axis_exact(
                    torch.randn(1, 3, generator=generator, dtype=torch.float64)
                )[0]
                if all(
                    float((candidate * prev).sum().abs()) < ALIGN_GREEDY_MAX
                    for prev in accepted
                ):
                    accepted.append(candidate)
                    slot += 1
                    tries = 0
                else:
                    tries += 1
            if len(accepted) < n:
                restarts += 1
        assert len(accepted) == n, (
            f"greedy packing failed after {GREEDY_RESTARTS} restarts"
        )
        out[b] = torch.stack(accepted)
    # step 2: off the diagonal, no pair of accepted axes is aligned past the
    # greedy ceiling; the diagonal self-pairs sit at cosine one by definition.
    cos = (out.unsqueeze(2) * out.unsqueeze(1)).sum(dim=-1).abs()
    off_diagonal = cos.masked_fill(torch.eye(n, dtype=torch.bool).unsqueeze(0), 0.0)
    assert bool((off_diagonal < ALIGN_GREEDY_MAX).all()), "packed axes must stay non-aligned"
    return out


def _construct_one(
    class_id: int, config: SpinGnnConfig, generator: torch.Generator
) -> Constellation:
    # step 2: resample one element until its label matches, up to the retry
    # budget; fail loud with the class id and the attempt count.
    piece = _task_b_element(class_id, config, generator)
    label = int(label_task_b(piece)[0])
    tries = 1
    while label != class_id and tries < CONSTRUCT_MAX_TRIES:
        piece = _task_b_element(class_id, config, generator)
        label = int(label_task_b(piece)[0])
        tries += 1
    assert label == class_id, (
        f"construct_task_b failed class {class_id} after {CONSTRUCT_MAX_TRIES} tries"
    )
    return piece


def construct_task_b(
    batch: int,
    class_id: int,
    config: SpinGnnConfig,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> Constellation:
    # step 1: build each element behind the retry contract, then concatenate.
    assert class_id in (0, 1, 2, 3), f"class_id must be in 0..3, got {class_id}"
    assert batch >= 1, "batch must be at least 1"
    pieces = [_construct_one(class_id, config, generator) for _ in range(batch)]

    # step 2: concatenate the per-element batches and move to the asked dtype.
    out = Constellation(
        x=torch.cat([p.x for p in pieces]).to(dtype),
        s=torch.cat([p.s for p in pieces]).to(dtype),
        u=torch.cat([p.u for p in pieces]).to(dtype),
        phi=torch.cat([p.phi for p in pieces]).to(dtype),
        omega=torch.cat([p.omega for p in pieces]).to(dtype),
        h=torch.cat([p.h for p in pieces]).to(dtype),
        v=torch.cat([p.v for p in pieces]).to(dtype),
        x_c=torch.cat([p.x_c for p in pieces]).to(dtype),
        h_c=torch.cat([p.h_c for p in pieces]).to(dtype),
    )
    assert_valid(out)
    assert bool(in_interior(out).all())
    labels = label_task_b(out)
    assert bool((labels == class_id).all()), "constructed batch must label as asked"
    if class_id == 0:
        assert bool((predicates(out).n_near <= TASK_B_NEAR_MIN - 1).all())
    if class_id == 1:
        assert bool((out.omega.abs() <= OMEGA_SLOW + 1e-9).all())
    if class_id == 2:
        assert bool((predicates(out).n_aligned_pairs >= 1).all())
    if class_id == 3:
        assert bool((predicates(out).n_aligned_pairs == 0).all())
    return out


def construct_task_b_flat(
    batch: int,
    config: SpinGnnConfig,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> tuple[Constellation, Tensor]:
    # step 1: the training prior is flat, so each class fills an equal quarter.
    assert batch % 4 == 0, f"batch must divide by 4, got {batch}"
    quarter = batch // 4
    parts = [
        construct_task_b(quarter, class_id, config, generator, dtype=dtype)
        for class_id in range(4)
    ]
    labels = torch.cat(
        [torch.full((quarter,), class_id, dtype=torch.int64) for class_id in range(4)]
    )

    # step 2: shuffle the elements so classes do not arrive in blocks.
    perm = torch.randperm(batch, generator=generator)
    out = Constellation(
        x=torch.cat([p.x for p in parts])[perm],
        s=torch.cat([p.s for p in parts])[perm],
        u=torch.cat([p.u for p in parts])[perm],
        phi=torch.cat([p.phi for p in parts])[perm],
        omega=torch.cat([p.omega for p in parts])[perm],
        h=torch.cat([p.h for p in parts])[perm],
        v=torch.cat([p.v for p in parts])[perm],
        x_c=torch.cat([p.x_c for p in parts])[perm],
        h_c=torch.cat([p.h_c for p in parts])[perm],
    )
    labels = labels[perm]
    assert_valid(out)
    assert bool((label_task_b(out) == labels).all()), "flat labels must match the labeler"
    assert bool((torch.bincount(labels, minlength=4) == quarter).all())
    return out, labels


def corrupt_for_task_a(
    c: Constellation, generator: torch.Generator
) -> tuple[Constellation, Tensor]:
    # step 1: targets come from the clean cloud: mean distance to the
    # controller, mean axis-to-line-of-sight cosine, and mean signed speed.
    assert_valid(c)
    rel = c.x - c.x_c.unsqueeze(1)
    d_ic = rel.norm(dim=-1).mean(dim=-1)
    line = unit_axis_exact(rel)
    align = (c.u * line).sum(dim=-1).mean(dim=-1)
    mean_omega = c.omega.mean(dim=-1)
    targets = torch.stack((d_ic, align, mean_omega), dim=-1)
    assert targets.shape == (c.x.shape[0], 3)

    # step 2: gaussian noise on positions folds back through the box; gaussian
    # noise on speeds clamps back into the legal range.
    x_noisy = project_to_box(c.x + NOISE_SIGMA_X * torch.randn_like(c.x, generator=generator))
    omega_noisy = torch.clamp(
        c.omega + NOISE_SIGMA_OMEGA * torch.randn_like(c.omega, generator=generator),
        -OMEGA_MAX,
        OMEGA_MAX,
    )
    noisy = Constellation(
        x=x_noisy,
        s=c.s,
        u=c.u,
        phi=c.phi,
        omega=omega_noisy,
        h=c.h,
        v=c.v,
        x_c=c.x_c,
        h_c=c.h_c,
    )
    assert_valid(noisy)
    return noisy, targets


def majority_spin_align(c: Constellation) -> Tensor:
    # step 1: pairs within TASK_D_DIST gate the vote; an aligned pair is one
    # whose cosine clears TASK_D_ALIGN.
    assert_valid(c)
    b, n = int(c.x.shape[0]), int(c.x.shape[1])
    disp = c.x.unsqueeze(2) - c.x.unsqueeze(1)
    dist = disp.norm(dim=-1)
    cos = (c.u.unsqueeze(2) * c.u.unsqueeze(1)).sum(dim=-1)
    eye = torch.eye(n, dtype=torch.bool, device=c.x.device).unsqueeze(0)
    near_pair = (dist < TASK_D_DIST) & ~eye
    aligned_pair = near_pair & (cos.abs() > TASK_D_ALIGN)

    # step 2: one is the answer exactly when more than half of the gated pairs
    # align; no gated pair at all means zero.
    n_near = near_pair.sum(dim=(-1, -2))
    n_aligned = aligned_pair.sum(dim=(-1, -2))
    out = torch.zeros(b, dtype=torch.int64, device=c.x.device)
    has_pairs = n_near > 0
    majority = n_aligned * 2 > n_near
    out = torch.where(has_pairs & majority, torch.ones_like(out), out)
    assert out.shape == (b,)
    assert out.dtype == torch.int64
    assert bool(((out == 0) | (out == 1)).all())
    return out


def rollout_hit(c: Constellation, steps: int = ROLLOUT_STEPS) -> Tensor:
    # step 1: free flight under reflect_into_box; the input stays untouched.
    assert_valid(c)
    assert steps >= 1, "steps must be at least 1"
    x = c.x.clone()
    delta = DT * c.omega.unsqueeze(-1) * c.u
    hit = torch.zeros(c.x.shape[0], c.x.shape[1], dtype=torch.bool, device=c.x.device)

    # step 2: accumulate a hit wherever a satellite passes within TAU_D.
    for _ in range(steps):
        x, delta = reflect_into_box(x, delta)
        dist = (x - c.x_c.unsqueeze(1)).norm(dim=-1)
        hit = hit | (dist < TAU_D)

    # step 3: the label is the capped count of distinct hitting satellites.
    count = hit.sum(dim=-1)
    out = torch.clamp(count, max=3).to(torch.int64)
    assert out.shape == (c.x.shape[0],)
    assert bool(((out >= 0) & (out <= 3)).all())
    return out


def class_frequencies(labels: Tensor, k: int = 4) -> Tensor:
    # step 1: a length-k histogram normalized to a simplex row.
    assert labels.ndim == 1, f"labels must be 1-d, got {tuple(labels.shape)}"
    assert labels.dtype == torch.int64, "labels must be int64"
    assert k >= 1, "k must be at least 1"
    counts = torch.bincount(labels, minlength=k).to(torch.float64)[:k]
    out = counts / counts.sum()
    assert out.shape == (k,)
    assert bool(((out >= 0.0) & (out <= 1.0)).all())
    return out

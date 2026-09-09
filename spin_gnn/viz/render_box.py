# the box renders: one constellation, one panel per layer, the symmetry gap
# curve. spec: docs/SPEC.md. every plot writes a png to a caller path and
# returns that path; nothing calls plt.show.
# flow:
# 1. plot_constellation draws the controller, the satellites, and their axes.
# 2. plot_layers draws one panel per layer boundary the model crosses.
# 3. plot_symmetry_gap drags one satellite off the interior and traces the
#    gap under one fixed rotation, step by step.

from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from mpl_toolkits.mplot3d.axes3d import Axes3D  # noqa: E402

from spin_gnn.constants import GAP_DRAG_STEPS, OMEGA_MAX, R_INTERIOR  # noqa: E402
from spin_gnn.constellation import assert_valid  # noqa: E402
from spin_gnn.geometry.frames import random_rotation  # noqa: E402
from spin_gnn.model.config import SpinGnnConfig  # noqa: E402
from spin_gnn.model.diagnostics import symmetry_gap  # noqa: E402
from spin_gnn.model.spin_gnn_gnn import SpinGnn, SpinGnnLayer  # noqa: E402
from spin_gnn.train.tasks_synthetic import sample_interior  # noqa: E402
from spin_gnn.types import Constellation  # noqa: E402


def _enlarge_position_step(model: SpinGnn) -> None:
    # step 1: the near-zero init hides the clipping effect; set the final layer
    # of every phi_x to weight 1.0 so the wall break is measurable. the same
    # move as the equivariance test, reimplemented here so production code
    # never imports a test module.
    for layer in model.layers:
        assert isinstance(layer, SpinGnnLayer), "model.layers must hold SpinGnnLayer"
        phi_x = layer.update.phi_x
        last = phi_x[-1]
        assert isinstance(last, torch.nn.Linear), "phi_x must end in a Linear"
        with torch.no_grad():
            last.weight.fill_(1.0)
            last.bias.fill_(0.0)


def _draw_box(ax: Axes3D) -> None:
    # step 1: the cube is its 12 edges, each a gray line between two corners.
    corners = torch.tensor(
        [[a, b, c] for a in (0.0, 1.0) for b in (0.0, 1.0) for c in (0.0, 1.0)]
    )
    edges = (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    )
    for i, j in edges:
        p = corners[[i, j]]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color="gray", linewidth=0.5)


def _draw_constellation(ax: Axes3D, c: Constellation, index: int, color_by: str) -> None:
    # step 1: the matplotlib stubs type the array arguments as scalars; draw
    # through an Any handle so the tensor values pass at runtime.
    draw: Any = ax
    x_c = c.x_c[index].detach().cpu()
    x = c.x[index].detach().cpu()
    u = c.u[index].detach().cpu()
    omega = c.omega[index].detach().cpu()
    sizes = c.s[index].detach().cpu()
    phi = c.phi[index].detach().cpu()
    # step 2: the controller is one dark point at the center of the box.
    draw.scatter([x_c[0]], [x_c[1]], [x_c[2]], color="black", s=200)

    # step 3: the color field picks phi or |omega| per the literal.
    field = phi if color_by == "phi" else omega.abs()
    cmap = "hsv" if color_by == "phi" else "viridis"
    draw.scatter(x[:, 0], x[:, 1], x[:, 2], c=field, cmap=cmap, s=sizes * 80)

    # step 4: quivers along u, length scaled by |omega| against the ceiling.
    tips = u * (0.1 * omega.abs() / OMEGA_MAX).unsqueeze(-1)
    draw.quiver(
        x[:, 0], x[:, 1], x[:, 2],
        tips[:, 0], tips[:, 1], tips[:, 2],
        length=1.0, normalize=False,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)


def plot_constellation(
    c: Constellation,
    index: int,
    path: Path,
    color_by: Literal["phi", "omega"] = "phi",
) -> Path:
    # step 1: draw the requested element of the batch into a fresh figure.
    assert_valid(c)
    assert 0 <= index < c.x.shape[0], f"index must sit in the batch, got {index}"
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    _draw_box(ax)
    _draw_constellation(ax, c, index, color_by)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    assert path.exists() and path.stat().st_size > 1000
    return path


def plot_layers(model: SpinGnn, c: Constellation, index: int, path: Path) -> Path:
    # step 1: one panel per layer boundary, encoder output through layer n.
    assert_valid(c)
    assert 0 <= index < c.x.shape[0], f"index must sit in the batch, got {index}"
    states = model.forward_layers(c)
    n_panels = len(states)
    fig = plt.figure(figsize=(4 * n_panels, 4))
    for k, state in enumerate(states):
        ax = fig.add_subplot(1, n_panels, k + 1, projection="3d")
        _draw_box(ax)
        _draw_constellation(ax, state, index, "phi")
        ax.set_title(f"layer {k}")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    assert path.exists() and path.stat().st_size > 1000
    return path


def symmetry_gap_series(
    model: SpinGnn,
    config: SpinGnnConfig,
    generator: torch.Generator,
) -> list[float]:
    # step 1: build the model at float64 so the rotation books are exact; the
    # interior start is then flat at the float floor and only the wall clip
    # lifts the curve.
    model = model.double()
    _enlarge_position_step(model)
    c = sample_interior(1, config, generator, dtype=torch.float64)
    rotation = random_rotation(1, generator)

    # step 2: drag satellite 0 from the interior radius out to radius 0.5 along
    # +x, and satellite 1 the matched radius along -x. at the start both are
    # interior so the step stays translationally covariant; at the end the two
    # satellites reach opposite walls and their inflated steps clip two faces.
    radii = torch.linspace(R_INTERIOR, 0.5, GAP_DRAG_STEPS, dtype=torch.float64)
    gaps: list[float] = []
    for radius in radii:
        x_new = c.x.clone()
        x_new[0, 0] = c.x_c[0] + torch.tensor(
            [float(radius), 0.0, 0.0], dtype=torch.float64
        )
        x_new[0, 1] = c.x_c[0] + torch.tensor(
            [-float(radius), 0.0, 0.0], dtype=torch.float64
        )
        dragged = Constellation(
            x=x_new, s=c.s, u=c.u, phi=c.phi, omega=c.omega,
            h=c.h, v=c.v, x_c=c.x_c, h_c=c.h_c,
        )
        gap = symmetry_gap(model, dragged, rotation)
        gaps.append(float(gap.max()))
    return gaps


def plot_symmetry_gap(
    model: SpinGnn,
    config: SpinGnnConfig,
    generator: torch.Generator,
    path: Path,
) -> Path:
    # step 1: the curve starts on the interior and ends half a box length out.
    gaps = symmetry_gap_series(model, config, generator)
    radii = torch.linspace(R_INTERIOR, 0.5, len(gaps), dtype=torch.float64)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(radii.numpy(), gaps, marker="o")
    ax.set_xlabel("radius of the dragged satellite")
    ax.set_ylabel("symmetry gap")
    ax.set_yscale("log")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    assert path.exists() and path.stat().st_size > 1000
    return path

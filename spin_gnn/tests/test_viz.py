# tests for the box renders. spec: docs/SPEC.md.
# flow:
# 1. a small config keeps the model cheap on cpu.
# 2. each example writes a png to a pytest tmp path and checks it is non-empty.
# 3. the layers example counts the panels; the gap example reads the curve
#    off the seam plot_symmetry_gap draws.

import matplotlib
import matplotlib.figure

matplotlib.use("Agg")
import torch  # noqa: E402

from spin_gnn.model.config import SpinGnnConfig  # noqa: E402
from spin_gnn.model.spin_gnn_gnn import build_model  # noqa: E402
from spin_gnn.train.tasks_synthetic import sample_interior  # noqa: E402
from spin_gnn.viz.render_box import (  # noqa: E402
    plot_constellation,
    plot_layers,
    plot_symmetry_gap,
    symmetry_gap_series,
)


def _small_config() -> SpinGnnConfig:
    return SpinGnnConfig(n=8, d=16, c=4, d_c=32, d_m=16, n_layers=2)


def test_plot_constellation_writes_png(generator: torch.Generator, tmp_path) -> None:
    config = _small_config()
    c = sample_interior(2, config, generator, dtype=torch.float64)
    out = plot_constellation(c, 0, tmp_path / "constellation.png")
    assert out.exists() and out.stat().st_size > 1000


def test_plot_layers_writes_n_layers_plus_1_panels(
    generator: torch.Generator, tmp_path, monkeypatch
) -> None:
    config = _small_config()
    model = build_model(config).double()
    c = sample_interior(2, config, generator, dtype=torch.float64)
    panels: list[int] = []
    real_subplot = matplotlib.figure.Figure.add_subplot

    def counting(self, *args, **kwargs):
        panels.append(1)
        return real_subplot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "add_subplot", counting)
    out = plot_layers(model, c, 0, tmp_path / "layers.png")
    assert out.exists() and out.stat().st_size > 1000
    assert len(panels) == config.n_layers + 1


def test_plot_symmetry_gap_curve_crosses_the_wall(tmp_path) -> None:
    # the n=8 small config with the six-wall drag starts at the float floor
    # and ends past the break floor (1e-4, four orders above the float floor,
    # matching test_equivariance); the seed fixes the draw.
    config = _small_config()
    generator = torch.Generator().manual_seed(0)
    gaps = symmetry_gap_series(build_model(config), config, generator)
    assert gaps[0] < 1e-6
    assert gaps[-1] > 1e-4
    out = plot_symmetry_gap(
        build_model(config), config, torch.Generator().manual_seed(0), tmp_path / "gap.png"
    )
    assert out.exists() and out.stat().st_size > 1000

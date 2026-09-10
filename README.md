# Spin_GNN

The weights are small on purpose. The foundation is the work.

A small SO(3)-equivariant graph network with one controller node and sixteen
spinning satellites in a unit cube, with every symmetry claim stated as a
proposition, proved, and property-tested.

## Status

2026-09-09, the `main` branch, 172 tests, `pytest -q` passes. The `full`
configuration has 526495 parameters.

## What this is

One controller node sits at the center of a unit cube and sixteen satellites
spin around it, each carrying a position, an axis, a phase, and a spin speed.
Four equivariant message-passing layers move invariant scalars and equivariant
vectors between the satellites and the controller. A set of predicates reads
the rigid state and feeds nothing back into the loss. The composition claim is
that stacking invariant packets with equivariant messages keeps the whole model
SO(3)-equivariant on the interior; every component is inherited from the cited
equivariant and set-function work.

## Propositions

| Prop | Claim | Test | Status |
|---|---|---|---|
| 1 | packet is SO(3)-invariant about the controller | test_invariance.py::test_packet_invariant | verified |
| 2 | message is SO(3)-equivariant | test_message.py | verified |
| 3 | full model is SO(3)-equivariant on the interior U | test_invariance.py, test_equivariance.py, diagnostics | verified at init |
| 4 | full model is exactly O-equivariant everywhere | test_equivariance.py | verified |
| 5 | phi is a scalar, not a geometric spin, in v0 | docs only | docs only |
| 6 | parity of omega is undeclared under reflections | docs only | out of scope |
| 7 | phase update is the exact constant-speed flow | test_phase_wrap.py::test_closed_form | verified |
| 8 | scalar path separates a distance-degenerate pair | test_expressivity.py | verified |

## Task B ablation

| run | params | heldout mean | heldout 95% CI | steps to 0.90 mean | steps to 0.90 95% CI | missing | sec per step |
|---|---|---|---|---|---|---|---|
| full | 526495 | 0.7502 | [0.7470, 0.7534] | null | null | 5 | 0.5277 |
| static | 526495 | 0.7570 | [0.7495, 0.7645] | null | null | 5 | 0.3641 |
| scalar_only | 526495 | 0.7494 | [0.7461, 0.7522] | null | null | 5 | 0.3572 |
| baseline | 526800 | 0.9202 | [0.9067, 0.9286] | 3925.0 | [3775.0, 4000.0] | 1 | 0.2050 |

paired steps-to-milestone difference (full - scalar_only): no paired seeds reached the milestone.

The paired interval does not exist: no full or scalar-only run reached the 0.90
milestone on any seed, so no seed pairs and the difference in steps to milestone
is undefined. It does not show that geometry updates help, hurt, or leave the
steps to milestone unchanged, because neither side produced a milestone to
compare. The milestone count is 16 missing of 20 seeded runs.

One panel per constructed class lives at `results/figures/class_0.png`,
`results/figures/class_1.png`, `results/figures/class_2.png`, and
`results/figures/class_3.png`. The per-layer message magnitudes are at
`results/figures/layers.png`.

## Class frequencies under uniform sampling

Uniform sampling places 0.91525 of draws in class 0, 0.0 in class 1, 0.08475 in
class 2, and 0.0 in class 3. The constructed generator is the dataset because
the uniform draw never produces classes 1 and 3, so the constructor builds the
near ring and the far ring explicitly to give Task B a flat prior.

## Where symmetry stops

Prop 4 says the box projection commutes only with isometries that map the cube
to itself, so the model is exactly equivariant under the 24 cube rotations and
breaks for a generic rotation once a satellite touches a wall. The gap curve at
`results/figures/symmetry_gap.png` shows the equivariance error grow as the
dragged satellite and its partner reach opposite walls.

## Limitations

phi is a scalar, not a geometric spin. Reflections are untested. Task B is a set
function, so the ablation measures efficiency, not necessity. The ablation runs
on CPU only. N is fixed at 16.

## Run

```bash
pip install -e ".[dev]"
pytest -q
python -m spin_gnn.train.ablate --seeds 5 --steps 4000 --out results
python -m spin_gnn.train.ablate --smoke
```

## Citations

The plan names thirteen references from `docs/MATH.md`, but that list is not in
the repository; only the works below are cited inline in the code and docs.

1. M. Boutin and G. Kemper, On reconstructing n-point configurations from the
   distribution of distances or areas, Advances in Applied Mathematics, 2004.
   The homometric line pair that witnesses Prop 8.
2. A. Pozdnyakov et al., Incompleteness of graph neural networks for points
   clouds in three dimensions, 2020. Three-dimensional homometric constructions
   noted as an alternative Prop 8 witness.
3. M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov, and
   A. Smola, Deep Sets, 2017. The width-matched baseline in the Task B ablation.

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

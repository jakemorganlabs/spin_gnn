# spin_gnn

Small SO(3)-equivariant GNN with one controller node and 16 spinning satellites
in a unit cube. Every symmetry claim is a proposition, proved, and
property-tested. PyTorch, CPU, under 1M params.

The model, readout, predicates, diagnostics, the Task B training loop, the
DeepSets baseline, and the seeded ablation are in place. Sessions 1 through 6
are built and measured; session 7 closes the claims ledger from `results/`.

## Status

Session 1 build: repository scaffold, constants, and the box, frame, and spin
geometry primitives, with property tests and gradcheck.

Session 2 build: the `Constellation` container with validation and the SO(3)
action about the controller, the invariant edge packet and controller self
packet, and the math docs. Prop 1, packet invariance, is property-tested under
Haar rotations. Claim C1 is verified.

Session 3 build: the frozen `SpinGnnConfig` with per-update flags, the
`IdentityEncoder`, the message layer, the gated satellite and controller
updates, and controller pooling. Prop 2, message equivariance, is
property-tested under Haar rotations, along with per-update equivariance, the
identical-tensor flag rule, and the position step bound at init. Claims C2 and
the session-3 half of C3 advance; C4 through C11 remain pending.

Session 4 build: `SpinGnnLayer` composes message pass, satellite update,
controller pool, and controller update; `SpinGnn` stacks `n_layers` of them
with no weight sharing and reads out through a continuous and a discrete head.
The predicate set returns booleans and counts under `no_grad`. Prop 3 invariance
and equivariance hold on the interior; Prop 4 is exact for all 24 cube
rotations everywhere and fails for a generic rotation at the walls, which is
the negative test. Prop 7 closes the phase integrator in closed form, and Prop
8 separates the Boutin-Kemper homometric pair. Claims C2 through C5, C7, and C8
are verified; C9 through C11 await training in sessions 5 and 6.

Session 5 build: the synthetic Task B generator constructs every class instead
of filtering, so the training prior is flat and the generator always returns.
Uniform sampling collapses the class mix to about 91.5 percent class 0, which
is why the constructor builds the near ring and the far ring explicitly. Tasks
A, D, and E sit beside the constructor, the three regularizers ride on a
`LossBreakdown`, and the plots render the constellation, one panel per layer,
and the symmetry-gap curve as the dragged satellite and its partner reach
opposite walls.

Session 6 build: the DeepSets baseline reads the fourteen scalar packet columns,
means over the off-diagonal edges, and matches the full parameter count within
ten percent. The training loop draws every batch from the constructed Task B
generator, evaluates a fixed held-out set every `EVAL_EVERY` steps, and records
the first step at which held-out accuracy reaches the milestone. The ablation
runs full, static, scalar-only, and baseline over the seed list, writes each run
to `results/task_b.json`, and folds the seeds into bootstrap intervals and a
paired full-versus-scalar-only difference in steps to milestone. The measured
numbers are in `results/`.

Session 6 result: the four configurations trained on constructed Task B over
five seeds, four thousand steps each, on CPU. The width-matched DeepSets
baseline beats every Spin_GNN variant on held-out accuracy and is the only
configuration that reaches the 0.90 milestone.

| run | params | heldout mean | heldout 95% CI | steps to 0.90 mean | steps to 0.90 95% CI | missing | sec per step |
|---|---|---|---|---|---|---|---|
| full | 526495 | 0.7502 | [0.7470, 0.7534] | null | null | 5 | 0.5277 |
| static | 526495 | 0.7570 | [0.7495, 0.7645] | null | null | 5 | 0.3641 |
| scalar_only | 526495 | 0.7494 | [0.7461, 0.7522] | null | null | 5 | 0.3572 |
| baseline | 526800 | 0.9202 | [0.9067, 0.9286] | 3925.0 | [3775.0, 4000.0] | 1 | 0.2050 |

The paired full-minus-scalar-only difference in steps to milestone is null:
neither run reaches the milestone on any seed, so no seed pairs. Every full run
ends with `within_bound_after` False: training pushed the per-layer position
step past the Prop 3 bound, while static and scalar-only stayed within it. The
three model variants are statistically indistinguishable from each other; the
scalar packet alone carries more Task B signal than the equivariant stack
extracts in this budget.

## Verify

```bash
pip install -e ".[dev]"
pytest -q
pyright spin_gnn
ruff check spin_gnn
```

## Limitations

The equivariant model loses to its own DeepSets baseline on the task it was
built to learn. On constructed Task B over five seeds the baseline reaches 0.920
held-out accuracy while the full model stalls near 0.750, and no model run
reaches the 0.90 milestone inside four thousand steps. The step bound proved at
initialization does not survive training: every full run ends with
`within_bound_after` False, so Prop 3 holds at init but not after optimization.
The claim that geometry updates move steps to milestone against the scalar-only
ablation is not supported here; neither side reaches the milestone, so the
paired difference is null. These are measured numbers, not aspirations, and they
bound what the project currently claims.

## Layout

- `spin_gnn/constants.py` holds every numeric constant.
- `spin_gnn/types.py` holds the `Constellation` dataclass and the tagged
  validation outcomes.
- `spin_gnn/constellation.py` validates, rotates, and measures constellations.
- `spin_gnn/geometry/` holds the box, frame, spin, and invariant primitives.
- `spin_gnn/model/` holds the config, the readout heads, the predicates, the
  stacked model, and the diagnostics.
- `spin_gnn/data/` holds the constructed degenerate pairs.
- `spin_gnn/tests/` holds the example and property tests.
- `docs/SPEC.md` is the layer spec. `docs/MATH.md` holds the propositions and
  proofs. `docs/CLAIMS.md` is the claims ledger.

  [![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

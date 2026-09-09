# spin_gnn

Small SO(3)-equivariant GNN with one controller node and 16 spinning satellites
in a unit cube. Every symmetry claim is a proposition, proved, and
property-tested. PyTorch, CPU, under 1M params.

The model, readout, predicates, and diagnostics are in place. The full README,
with the claims ledger and the measured numbers, is written in session 7.

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

## Verify

```bash
pip install -e ".[dev]"
pytest -q
pyright spin_gnn
ruff check spin_gnn
```

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

# spin_gnn

Small SO(3)-equivariant GNN with one controller node and 16 spinning satellites
in a unit cube. Every symmetry claim is a proposition, proved, and
property-tested. PyTorch, CPU, under 1M params.

The container, the invariant edge packet, and the math docs are in place. The
full README, with the claims ledger and the measured numbers, is written in
session 7.

## Status

Session 1 build: repository scaffold, constants, and the box, frame, and spin
geometry primitives, with property tests and gradcheck.

Session 2 build: the `Constellation` container with validation and the SO(3)
action about the controller, the invariant edge packet and controller self
packet, and the math docs. Prop 1, packet invariance, is property-tested under
Haar rotations. Claim C1 is verified; C2 through C11 are pending.

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
- `spin_gnn/tests/` holds the example and property tests.
- `docs/SPEC.md` is the layer spec. `docs/MATH.md` holds the propositions and
  proofs. `docs/CLAIMS.md` is the claims ledger.

# spin_gnn

Small SO(3)-equivariant GNN with one controller node and 16 spinning satellites
in a unit cube. Every symmetry claim is a proposition, proved, and
property-tested. PyTorch, CPU, under 1M params.

The geometry foundation is in place. The full README, with the claims ledger
and the measured numbers, is written in session 7.

## Status

Session 1 build: repository scaffold, constants, and the box, frame, and spin
geometry primitives, with property tests and gradcheck.

## Verify

```bash
pip install -e ".[dev]"
pytest -q
pyright spin_gnn
ruff check spin_gnn
```

## Layout

- `spin_gnn/constants.py` holds every numeric constant.
- `spin_gnn/geometry/` holds the box, frame, and spin primitives.
- `spin_gnn/tests/` holds the example and property tests.

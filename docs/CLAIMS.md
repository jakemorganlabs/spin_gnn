# CLAIMS: the ledger

Every claim the project makes, where it is proved or measured, and its
status. A row opens unresolved and dies as `refuted`. A row closes
only when its test or script passes in CI.

| id | claim | where proved or measured | test or script | status |
|---|---|---|---|---|
| C1 | packet is SO(3)-invariant about C | Prop 1 | spin_gnn/tests/test_invariance.py::test_packet_invariant | verified |
| C2 | message is SO(3)-equivariant | Prop 2 | spin_gnn/tests/test_message.py | verified |
| C3 | full model is SO(3)-equivariant on interior U | Prop 3 | spin_gnn/tests/test_invariance.py, test_equivariance.py, diagnostics | verified; step bound measured at init; re-measured after training in session 6 |
| C4 | full model is exactly O-equivariant everywhere | Prop 4 | spin_gnn/tests/test_equivariance.py::test_octahedral_exact_on_wall_touching_batch | verified |
| C5 | full model is NOT generically equivariant at walls | Prop 4 | spin_gnn/tests/test_equivariance.py::test_generic_breaks | verified |
| C6 | phi is a scalar, not a geometric spin, in v0 | Prop 5 | docs only | docs only |
| C7 | phase update is the exact constant-speed flow | Prop 7 | spin_gnn/tests/test_phase_wrap.py::test_closed_form | verified |
| C8 | scalar path separates a distance-degenerate pair | Prop 8 | spin_gnn/tests/test_expressivity.py | verified |
| C9 | Task B held-out accuracy at least 0.95, 5 seeds, CI reported | results | results/task_b.json | failed; measured 0.7502 |
| C10 | geometry updates change steps to 0.90 versus the scalar-only model (vector state removed, vector-derived scalars kept) | results | results/task_b.json | not shown; no paired seeds reached the milestone, interval undefined |
| C11 | everything reproduces from one command and one seed | run | spin_gnn/tests/test_determinism.py, spin_gnn/train/ablate.py | verified |

## How a row closes

1. Write the test or the script that the row names.
2. Run it. Numbers that depend on training are written under `results/`.
3. Flip the status and name the commit that flipped it.

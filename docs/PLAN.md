# PLAN: the session ladder, where it stands, and the end goal

This file is the memory between sessions. Each session opens by reading it,
closes by updating it. The end goal sits at the bottom and does not move.

## The ladder so far

| session | what landed | commit |
|---|---|---|
| 1 | box, frame, and spin primitives with property tests | fe161fe |
| 2 | constellation container, invariant edge packet, MATH.md | 4403c46 |
| 3 | message pass, node updates, controller pool behind config flags | cf33a81 |
| 4 | four stacked layers, readout, predicates, symmetry claims proved | 5f5daee |
| 5 | constructed Task B generator, tasks A, D, E, losses, plots | c876b51 |
| 6 | training loop, DeepSets baseline, seeded ablation | c872500 |
| 7 | measured numbers, claims ledger closed, license, figures | 945316d |
| 8 | the model fix: basis expansion, max and attention aggregation, PaiNN block, layer norm, schedule, cached stream | this branch |

## What session 7 found

The session 6 model scored 0.7502 held-out on Task B against 0.9202 for the
width-matched DeepSets baseline, and no seed of any Spin_GNN configuration
reached the 0.90 milestone. C9 failed and C10 could not be shown.

## What session 8 diagnosed

A 300-step run with the per-class confusion matrix showed classes 0 and 1
learned inside 150 steps and classes 2 and 3 confused at chance for the
whole run. Class 2 differs from class 3 by exactly one aligned axis pair
among the 120 edges. Mean aggregation over 16 neighbors hands that one
edge a weight of 1/16 at the satellite and then averages it again at the
controller, so the signal never clears the noise floor. Two smaller faults
sat beside it: the training loop ran the model forward twice per step, and
`clamp_speed` folded signed speeds into [0, OMEGA_MAX] against the SPEC.

## What session 8 changed

Every change keeps Props 1 through 8; Prop 9 in MATH.md proves the new
maps and the property tests cover them.

1. Gaussian grids over the pair distance and the axis alignment columns
   (SchNet), so a threshold is a linear read of a bump.
2. Four aggregators, mean, max, the per-channel softmax aggregator with a
   learned temperature (DeeperGCN), and softmax attention, at the
   satellite and at the controller (principal neighbourhood aggregation,
   equivariant transformer), plus a direct pool of every satellite edge
   into the controller. The vector message aggregates by mean plus
   attention. The first cut used only mean, max, and attention and still
   fused classes 2 and 3 at 300 steps; the hard max passes gradient
   through one edge per channel, which is why the softmax aggregator and
   the edge pool were added. A second 1000-step probe with those still
   sat at 0.74 by step 400, so the raw invariant edge features are now
   pooled by max and mean before any mixing, at the satellite and at the
   controller, which makes the one-pair threshold a linear read.
3. The PaiNN gated equivariant block: channel mixing on V and per-channel
   invariants `||U V||, <U V, W V>, ||M||, <V, M>` feeding the scalar path.
4. Layer norm on the scalar inputs of every MLP and on the readout heads.
5. The encoder reads `d_iC` and its gaussian grid, and `log s`.
6. One forward per step, AdamW at 1e-3 with warmup and cosine decay,
   gradient clipping at 1.0.
7. A cached, bitwise-deterministic Task B stream per seed under
   `results/cache/`, shared by the four runs of that seed.
8. `clamp_speed` keeps the sign.

## Next session (9): measure and publish

1. Run `python -m spin_gnn.train.ablate --seeds 5 --steps 4000 --out results`
   on the VPS (about 6 hours on 4 CPU cores; run it under tmux or nohup).
   If it is already running or done when the session opens, read
   `results/task_b.json` first.
2. Regenerate the README table from `results/summary.md` and the numbers
   from `scripts/readme_numbers.py`; update `docs/CLAIMS.md` rows C9 and
   C10 from the measured mean and the paired interval; the readme test
   enforces both.
3. Re-measure the post-training step bound (`max_step_after`) and record it
   in MATH.md under Prop 3.
4. If C9 still fails, the next levers in order: widen `D_MESSAGE` to 128,
   raise `N_RBF_COS` to 32, add a second attention head, train 8000 steps.
   Change one thing per run and keep the seed-0 confusion matrix in the log.

## Session 10: the geometric updates must earn their place

C10 asks whether the geometric updates change the steps to the milestone.
Task B is a set function, so the honest test needs a task where moving the
satellites matters: Task E (rollout hit) and Task D (majority spin
alignment) already have labelers in `tasks_synthetic.py`. Train the
continuous head on Task A and the discrete head on D and E with the same
four-way ablation, and publish the paired intervals.

## Session 11: close the v0 limitations

1. Prop 5: give phi a transported reference frame so the phase difference
   is geometric; prove and test the new packet columns.
2. Prop 6: declare the parity of omega and test the model under O_h.
3. Variable N with padding masks; the aggregators already tolerate it.
4. A GPU run of the ablation to lift the step budget.

## End goal

A small SO(3)-equivariant graph network where every symmetry claim is a
proposition with a proof and a property test, every published number is
measured by one command from one seed, and the equivariant stack beats the
width-matched set baseline on the constructed tasks while the geometric
updates are shown to shorten training on a task that needs them. The
weights stay small on purpose; the foundation is the work.

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
| 8 | the model fix: basis expansion, multi-aggregation, PaiNN block, layer norm, schedule, cached stream; measured 1.0000 on Task B | fb228da, c088ed2 |

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
9. The controller seed starts at unit scale. The first canonical run hit
   0.962 held-out at step 100 and then spiked to loss 1.7 at peak learning
   rate; per-module gradient norms put the spike on the 0.02-scale seed
   and the first controller update, because the layer norm that reads the
   seed multiplies gradients by one over its spread. At unit scale the
   worst gradient norm over 250 steps fell from 140 to 29 and the loss
   reached 0.017 by step 250 on the 4000-step schedule.

## Session 8 also measured and published

The canonical run finished on 2026-09-11 after 10 hours 20 minutes on the
VPS. Every Spin_GNN configuration reached 1.0000 held-out on every seed
with the milestone between step 100 and step 200; the baseline reached
0.9554 at about step 2170. C9 closed as verified. C10 stays not shown:
the paired interval over five seeds is [-50.0, 40.0]. The trained full
model steps past the Prop 3 bound on every seed (max_step 0.097 to 0.578
against 0.0375), recorded in MATH.md. The results, summary, and figures
under `results/` are the session 8 measurement.

## Next session (9): the geometric updates must earn their place

C10 asks whether the geometric updates change the steps to the milestone.
Task B is a set function, so the honest test needs a task where moving the
satellites matters: Task E (rollout hit) and Task D (majority spin
alignment) already have labelers in `tasks_synthetic.py`. Train the
continuous head on Task A and the discrete head on D and E with the same
four-way ablation, and publish the paired intervals.

## Session 10: close the v0 limitations

1. Prop 5: give phi a transported reference frame so the phase difference
   is geometric; prove and test the new packet columns.
2. Prop 6: declare the parity of omega and test the model under O_h.
3. Variable N with padding masks; the aggregators already tolerate it.
4. A GPU run of the ablation to lift the step budget.
5. Prop 3 after training: bound the learned position step by construction
   (a tanh on phi_x scaled to MARGIN / N_LAYERS) so within_bound holds
   after training, then re-measure max_step_after.

## End goal

A small SO(3)-equivariant graph network where every symmetry claim is a
proposition with a proof and a property test, every published number is
measured by one command from one seed, and the equivariant stack beats the
width-matched set baseline on the constructed tasks while the geometric
updates are shown to shorten training on a task that needs them. The
weights stay small on purpose; the foundation is the work.

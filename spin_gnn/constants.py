# every numeric constant for the project lives here, and only here.
# flow:
# 1. declare float, int, and shape constants in SCREAMING_SNAKE_CASE.
# 2. comment each constant with its unit and its reason.
# 3. later modules import from here so one file holds every number.

EPS: float = 1e-8  # dimensionless; stabilizes norms and divisions against exact zeros
PI: float = 3.141592653589793  # radians; half turn
TWO_PI: float = 6.283185307179586  # radians; full turn, used by angle wrapping
DT: float = 0.25  # seconds per integration step; four steps per second of simulated spin
OMEGA_MAX: float = 2.0  # radians per second; spin speed ceiling for class balance
SIZE_MIN: float = 1e-3  # normalized units; smallest visible satellite extent
SIZE_MAX: float = 2.0  # normalized units; largest satellite extent that stays in the box
MARGIN: float = 0.15  # box-length units; keeps satellites away from walls at spawn
BOX: tuple[float, float, float] = (1.0, 1.0, 1.0)  # length units; the unit cube extent per axis
R_INTERIOR: float = 0.35  # box-length units; 0.5 - MARGIN, written out so it is greppable
N_SATELLITES: int = 16  # count; satellites orbiting the single controller node
D_SCALAR: int = 64  # channels; width of the invariant scalar node state
C_VECTOR: int = 16  # channels; number of equivariant 3-vectors per node
D_CONTROLLER: int = 128  # channels; width of the controller node state
D_MESSAGE: int = 64  # channels; width of edge messages
N_LAYERS: int = 4  # count; message passing layers in the model
TAU_D: float = 0.25  # box-length units; distance predicate threshold
TAU_U: float = 0.95  # cosine; axis non-alignment threshold, greedy class 3 packing needs 0.95
TAU_OMEGA: float = 0.5  # radians per second; spin speed predicate threshold
TAU_W: float = 0.08  # normalized units; wall-distance predicate threshold
TAU_PHI: float = 0.30  # radians; phase difference predicate threshold
INIT_SMALL: float = 0.01  # weight scale; small init keeps the step bound provable at start
SEED: int = 0  # index; master seed so every experiment reproduces exactly

# session 5: task, loss, and training constants
W_UNIT: float = 0.1  # loss weight; pulls axes back to unit length
W_BOX: float = 0.1  # loss weight; pushes positions back inside the box
W_SPIN: float = 0.01  # loss weight; keeps speeds below the ceiling
LR: float = 1e-3  # inverse steps; AdamW peak learning rate, reached after WARMUP_STEPS
WEIGHT_DECAY: float = 1e-4  # dimensionless; Adam L2 pull
BATCH_SIZE: int = 64  # examples; per optimization step
TRAIN_STEPS: int = 4000  # steps; total optimization budget
HELD_OUT_SIZE: int = 2000  # examples; fixed evaluation set
N_SEEDS: int = 5  # runs; ablation repetitions for confidence intervals
ACC_TARGET: float = 0.95  # fraction; Task B accuracy the model must beat
ACC_MILESTONE: float = 0.90  # fraction; Task B midpoint progress marker
TASK_B_NEAR_MIN: int = 3  # count; class 0 has fewer near satellites than this
OMEGA_MEAN_THRESHOLD: float = 0.4  # radians per second; class 1 mean-speed ceiling
OMEGA_SLOW: float = 0.3  # radians per second; every class 1 speed stays below this
OMEGA_FAST_MIN: float = 0.8  # radians per second; class 2 and 3 speeds start here
ALIGN_GREEDY_MAX: float = 0.9  # cosine; class 3 greedy packing acceptance ceiling
NEAR_R_MIN: float = 0.05  # box-length units; smallest sampled radius off the controller
NEAR_PAD: float = 0.02  # box-length units; keeps constructed radii off the TAU_D edge
NEAR_K_MAX: int = 8  # count; upper bound on the constructed near-satellite count
CONSTRUCT_MAX_TRIES: int = 100  # resamples; per-element constructor retry budget
GREEDY_AXIS_TRIES: int = 500  # candidates; per-slot greedy packing budget
GREEDY_RESTARTS: int = 20  # restarts; whole-set greedy packing budget
TASK_D_DIST: float = 0.4  # box-length units; pair distance gate for task D
TASK_D_ALIGN: float = 0.5  # cosine; pair alignment gate for task D
ROLLOUT_STEPS: int = 4  # steps; task E free-flight horizon
NOISE_SIGMA_X: float = 0.05  # box-length units; task A position noise width
NOISE_SIGMA_OMEGA: float = 0.2  # radians per second; task A speed noise width
GAP_DRAG_STEPS: int = 20  # points; drag resolution for the symmetry gap curve

# session 6: evaluation and bootstrap constants
EVAL_EVERY: int = 50  # steps; held-out cadence, fine enough to place a milestone near step 200
HELD_OUT_SEED_OFFSET: int = 100000  # index; shifts the held-out draw off the training stream
BOOTSTRAP_N: int = 10000  # resamples; bootstrap replicates per confidence interval
CI_LOW: float = 0.025  # fraction; lower bootstrap quantile
CI_HIGH: float = 0.975  # fraction; upper bootstrap quantile
PARAM_MATCH_TOL: float = 0.10  # fraction; baseline width matches the full parameter count
SMOKE_STEPS: int = 20  # steps; the smoke training budget keeps CI under two minutes
MAX_STEP_BATCHES: int = 8  # batches; interior draws for the post-training step bound

# session 8: basis expansion, aggregation, and training schedule constants
N_RBF_DIST: int = 24  # count; gaussian centers over the pair distance column
RBF_DIST_MAX: float = 1.75  # box-length units; just above the cube diagonal sqrt(3)
N_RBF_COS: int = 32  # count; alignment grid on [-1, 1], width 0.065 resolves TAU_U
N_RBF_CONTROLLER: int = 16  # count; gaussian centers over the controller distance
RBF_CONTROLLER_MAX: float = 0.9  # box-length units; just above the center-to-corner distance
WARMUP_STEPS: int = 200  # steps; linear learning-rate warmup before the cosine decay
LR_MIN_FRAC: float = 0.05  # fraction; the cosine schedule floor as a fraction of LR
GRAD_CLIP: float = 1.0  # norm; global gradient-norm ceiling per optimizer step
EVAL_CHUNK: int = 500  # examples; held-out forward chunk so the packet fits in memory
CACHE_DIR: str = "results/cache"  # path; cached Task B streams live here, relative to the repo
SOFTMAX_AGG_INIT: float = 1.0  # inverse temperature; softmax aggregator starts as a soft mean
H_C_INIT_STD: float = 1.0  # scale; unit-scale controller seed keeps the layer norm Jacobian O(1)

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

# SPEC: the Spin_GNN layer in one place

This file is the single source for the layer. MATH.md carries the proofs.
CLAIMS.md carries the ledger.

## Objects

The world is a batch of B constellations. Each constellation has one
controller C and N = 16 satellites in the unit cube. A satellite i carries:

| field | shape | meaning |
|---|---|---|
| `x` | (B, N, 3) | position in the box |
| `s` | (B, N) | size, positive |
| `u` | (B, N, 3) | unit spin axis |
| `phi` | (B, N) | spin phase in (-pi, pi], an invariant scalar in v0 |
| `omega` | (B, N) | signed spin speed in [-OMEGA_MAX, OMEGA_MAX] |
| `h` | (B, N, D_SCALAR) | invariant scalar state |
| `v` | (B, N, 3, C_VECTOR) | equivariant vector state |
| `x_c` | (B, 3) | controller position |
| `h_c` | (B, D_CONTROLLER) | controller state |

The container is the frozen dataclass `Constellation` in
`spin_gnn/types.py`. Every transform returns a new instance.

## Constants

Every number lives in `spin_gnn/constants.py`. The ones this spec uses:

| name | value | role |
|---|---|---|
| `BOX` | (1.0, 1.0, 1.0) | the unit cube extent |
| `MARGIN` | 0.15 | spawn clearance from the walls |
| `R_INTERIOR` | 0.35 | interior radius, 0.5 - MARGIN |
| `EPS` | 1e-8 | norm and division stabilizer |
| `DT` | 0.25 | seconds per integration step |
| `OMEGA_MAX` | 2.0 | spin speed ceiling |
| `SIZE_MIN`, `SIZE_MAX` | 1e-3, 2.0 | size range |
| `D_SCALAR` | 64 | scalar state width d |
| `C_VECTOR` | 16 | vector state channels c |
| `D_CONTROLLER` | 128 | controller state width d_c |
| `D_MESSAGE` | 64 | message width |
| `N_LAYERS` | 4 | message passing layers |
| `TAU_D`, `TAU_U`, `TAU_OMEGA`, `TAU_W`, `TAU_PHI` | 0.25, 0.95, 0.5, 0.08, 0.30 | predicate thresholds |
| `INIT_SMALL` | 0.01 | init weight scale for the step bound |
| `N_RBF_DIST`, `RBF_DIST_MAX` | 24, 1.75 | gaussian grid over the pair distance |
| `N_RBF_COS` | 32 | gaussian grid over the axis alignment in [-1, 1] |
| `N_RBF_CONTROLLER`, `RBF_CONTROLLER_MAX` | 16, 0.9 | gaussian grid over the controller distance |
| `SOFTMAX_AGG_INIT` | 1.0 | starting inverse temperature of the softmax aggregators |
| `LR`, `WARMUP_STEPS`, `LR_MIN_FRAC`, `GRAD_CLIP` | 1e-3, 200, 0.05, 1.0 | the training schedule |
| `SEED` | 0 | master seed |

## Tensor shapes

The edge packet is (B, N, N, 14 + 2 d) with columns in `PacketColumn`
order: 14 scalars, then `h_i`, then `h_j`. The scalar columns are
`D, LOG_D, RHO, SIGMA, ALPHA, SIN_DPHI, COS_DPHI, DOMEGA, OMEGA_PROD,
BETA_I_J, BETA_J_I, COS_PSI_C, U_I_TO_C, U_J_TO_C`. The U columns are the
inner product of a satellite axis with its sight line to the controller;
both are inner products of co-rotating vectors, so Prop 1 covers them.
The diagonal rows (i == j) are computed and left for the caller to mask.

The self packet is (B, N, 14 + 2 d). It follows the convention
`phi_c = 0`, `omega_c = 0`, `u_c` equal to the unit vector from `x_c` to
`x_i`, `s_c = 1`, and the caller's projection of `h_c` in the `h_j` slot.

## The layer in one place

```
r_ij = x_i - x_j;  d_ij = ||r_ij||_eps;  r_hat_ij = r_ij / d_ij
alpha_ij = <u_i,u_j>;  dphi_ij = atan2(sin(phi_i - phi_j), cos(phi_i - phi_j));  beta_{i<-j} = <u_i, r_hat_ij>
L_j = s_j omega_j u_j
e_ij = pack(...)
f_ij = [e_ij, rbf(d_ij; 0, RBF_DIST_MAX, N_RBF_DIST), rbf(alpha_ij; -1, 1, N_RBF_COS)]
m_ij = phi_m(f_ij);  a_ij = softmax_j(w_a . m_ij)   over the N neighbors of i
M_ij = V_j (.) g_v + r_hat_ij (x) g_r + u_j (x) g_u + L_j (x) g_L
soft_j(m; beta) = sum_j softmax_j(beta (.) m_ij) (.) m_ij      per channel, beta learned
f_raw_ij = the first 14 + N_RBF_DIST + N_RBF_COS columns of f_ij
m_i = W_agg [mean_j m_ij, max_j m_ij, soft_j(m_ij; beta), sum_j a_ij m_ij, max_j f_raw_ij, mean_j f_raw_ij]
M_i = mean_j M_ij + sum_j a_ij M_ij
q_i = [||U V_i||_c, <U V_i, W V_i>_c, ||M_i||_c, <V_i, M_i>_c, ||M_i||_F]
h_i     <- h_i + phi_h(LN(h_i), m_i, q_i, s_i, log s_i, omega_i/OMEGA_MAX, sin phi_i, cos phi_i, d_iC)
V_i     <- V_i + sigmoid(g(LN(h_i))) (.) (M_i W_m + V_i W_s)
s_i     <- clamp(softplus(softplus_inv(s_i) + phi_s(LN(h_i))))
u_i     <- normalize(u_i + (M_i 1_c) tanh(phi_axis(LN(h_i))))
omega_i <- clip(omega_i + phi_omega(LN(h_i)), -OMEGA_MAX, OMEGA_MAX)
phi_i   <- wrap(phi_i + omega_i DT)
x_i     <- Pi_B(x_i + mean_j r_hat_ij phi_x(m_ij)/(d_ij + 1))
m_C     <- W_agg_C [mean_i m_iC, max_i m_iC, soft_i(m_iC; beta_C), sum_i softmax_i(w_a . m_iC) m_iC,
                    mean_ij m_ij, soft_ij(m_ij; beta_E), max_ij f_raw_ij, mean_ij f_raw_ij]
                    the last four over every satellite edge
h_C     <- h_C + phi_C(LN(h_C), m_C, pool({P_i}))
y = psi_cont(LN(h_C));  z = psi_disc(LN(h_C));  pi = predicates(P, C)
```

`rbf(x; lo, hi, k)` is the gaussian grid of `spin_gnn/geometry/basis.py`:
k centers evenly spaced on [lo, hi] with width equal to the spacing. `U`,
`W`, `W_m`, and `W_s` are c by c maps on the channel axis of V. `LN` is a
layer norm over the scalar width. The encoder seeds h from
`[s, log s, omega/OMEGA_MAX, sin phi, cos phi, d_iC, rbf(d_iC; 0, RBF_CONTROLLER_MAX, N_RBF_CONTROLLER)]`.

Satellites receive a controller message through the same MLP as satellite
edges, built from `self_packet`. The neighbor count is N: the N-1 other
satellites and the controller, which fills the last attention slot. The
controller pool attends over satellites with logits from `[LN(h_i), s_i,
omega_i/OMEGA_MAX, sin phi_i, cos phi_i, d_iC]`.

## Training

AdamW at peak LR with WEIGHT_DECAY, linear warmup over WARMUP_STEPS then
cosine decay to LR_MIN_FRAC of the peak, gradient norm clipped at GRAD_CLIP,
one forward per step. The Task B stream for a seed is drawn once and cached
under CACHE_DIR, so every run of that seed sees bitwise the same batches.

## Predicates

The discrete head reads predicates of the constellation:

| predicate | threshold | meaning |
|---|---|---|
| close pair | `TAU_D` | two satellites within 0.25 |
| aligned axes | `TAU_U` | cosine between axes above 0.95 |
| fast spin | `TAU_OMEGA` | `abs(omega)` above 0.5 |
| near wall | `TAU_W` | wall distance below 0.08 |
| phase lock | `TAU_PHI` | `abs(dphi)` below 0.30 |

Every number the README publishes is measured by `spin_gnn/train/ablate.py`.

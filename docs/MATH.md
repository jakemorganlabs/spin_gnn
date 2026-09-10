# MATH: definitions, propositions, proofs, and their tests

Each proposition has a Statement, a Proof, and a Test line. Tests that
arrive in later sessions are listed with their future path and read
`pending`. Claims C1 through C13 live in CLAIMS.md.

## 1. Definitions

The box B is the axis-aligned cube with extent BOX = (1, 1, 1). The
controller C is a single node at position x_c. A satellite is the tuple
(x_i, s_i, u_i, phi_i, omega_i, h_i, V_i) with x_i in B, s_i > 0, u_i a
unit vector, phi_i in (-pi, pi], omega_i in [-OMEGA_MAX, OMEGA_MAX], h_i
in R^d, and V_i in R^(3 x c).

Def 1 (the SO(3) action rho_R). For R in SO(3):

```
x_i -> x_c + R (x_i - x_c)
u_i -> R u_i
V_i -> R V_i        (on the 3-axis)
identity on s, phi, omega, h, h_c, x_c
```

Def 2 (the interior region U). Every satellite lies within R_INTERIOR of
x_c, with x_c at the box center. R_INTERIOR = 0.5 - MARGIN = 0.35, so no
point of U can reach a wall.

## 2. Prop 1: packet invariance

Statement. For every constellation P and every R in SO(3),
`e_ij(rho_R P) = e_ij(P)`.

Proof. Read the packet column by column. Each column is an inner product
of two vectors that both transform by R, a norm of such a vector, or a
scalar that does not transform. Inner products with a shared orthogonal
factor are unchanged: `<R a, R b> = <a, b>`. Norms are unchanged by the
same fact with a = b. The scalars s, phi, omega, h are fixed by Def 1.
The sight-line columns U_I_TO_C and U_J_TO_C are inner products of u with
the unit vector toward x_c, and both factors co-rotate.
Test: `spin_gnn/tests/test_invariance.py::test_packet_invariant`.

## 3. Prop 2: message equivariance

Statement. For every constellation P and every R in SO(3),
`M_ij(rho_R P) = R M_ij(P)`.

Proof. M_ij is a sum of four terms. Each term is a basis 3-vector that
transforms by R (V_j, r_hat_ij, u_j, L_j = s_j omega_j u_j) multiplied by
a gate that is a function of e_ij. The gates are invariant by Prop 1. So
each term transforms by R, and the sum does too.
Test: `spin_gnn/tests/test_message.py::test_message_equivariance`. Status: verified.

## 4. Prop 3: layer equivariance on U

Statement. For P in U and per-update step sizes bounded by MARGIN over
N_LAYERS layers, `project_to_box` is the identity on every step, every
update composes an invariant scalar map with an equivariant vector map,
and the layer commutes with rho_R.

Proof sketch. The axis update is bounded by tanh, so its step is at most
`abs(phi_axis)`. The speed update is bounded by clip. The position update
is bounded by the near-zero init INIT_SMALL and the 1/(d + 1) factor.
Projections never fire while satellites stay off the walls, so the layer
is a composition of maps that each commute with rho_R. The bound is
measured by `diagnostics.measure_max_step`, not assumed.
Test: `spin_gnn/tests/test_invariance.py::test_model_invariant_outputs_under_haar`
and `spin_gnn/tests/test_equivariance.py::test_model_equivariant_fields_on_interior`.
Status: verified.
Post-training step bound, seed 0, session 6 model: max_step 0.2328, within_bound False. The session 8 re-measurement is recorded in results/task_b.json under max_step_after.

## 5. Prop 4: what the box breaks

Statement. `project_to_box` commutes with an isometry g fixing x_c
exactly when g maps B to itself. For the cube about its center that group
is O_h (48 elements), or O (24 elements) without reflections.

Consequence. The model is exactly O-equivariant everywhere,
approximately SO(3)-equivariant on U, and not equivariant for generic
rotations of clouds that touch a wall.
Test: `spin_gnn/tests/test_equivariance.py::test_octahedral_exact_on_wall_touching_batch`
and `spin_gnn/tests/test_equivariance.py::test_generic_breaks`. Status: verified.

## 6. Prop 5: phi is a scalar in v0

Statement. A rotation angle about u_i is meaningful only relative to a
reference vector orthogonal to u_i. Comparing phi_i and phi_j across
different axes needs a transport convention, and v0 defines none.

So phi_i is an invariant scalar that evolves by `phi <- phi + omega DT`.
The phase difference column SIN_DPHI and COS_DPHI is informative only to
the extent that the axes are aligned; nothing in v0 claims otherwise.
Limitation: the model cannot express twist relative to a transported
frame.

## 7. Prop 6: parity

Statement. Under a reflection, u is a polar vector and L = s omega u is
polar unless omega is a pseudoscalar. Physical angular momentum is axial.
v0 restricts the group to SO(3), so the question of whether omega carries
pseudoscalar sign is deferred.
Test: none. The reflection group is out of scope for v0.

## 8. Prop 7: the phase integrator is exact

Statement. With omega fixed, L steps of the phase update give
`phi_L = wrap(phi_0 + L DT omega)`. No integration error accumulates.
Test: `spin_gnn/tests/test_phase_wrap.py::test_closed_form`. Status: verified.

## 9. Prop 8: expressivity lower bound

Statement. The packet contains distances and angles, so the scalar path
separates at least one pair of point clouds whose pairwise-distance
multisets are identical.

Witness. The homometric line sets `{0,1,4,10,12,17}` and
`{0,1,8,11,13,17}` of Boutin and Kemper (2004) have the same distance
multiset `{1,...,13,16,17}`. Embedded on a line through the controller
with N = 6, the angle columns of the packet differ between the two sets,
so the scalar path separates them exactly, in integers. The 3D homometric
constructions of Pozdnyakov et al. would also serve; the line pair is
chosen because it needs no numeric search.
Test: `spin_gnn/tests/test_expressivity.py::test_model_separates_the_homometric_pair`.
Status: verified. This is a lower bound, not a characterization.

## 9b. Prop 9: aggregation, basis expansion, and channel mixing keep the symmetry

Statement. Let e_ij be the invariant packet of Prop 1 and M_ij the
equivariant edge message of Prop 2. Then each of the following is invariant
(scalars) or equivariant (vectors) under rho_R:

1. `rbf(e_ij[k]; lo, hi, K)` for any packet column k.
2. `max_j m_ij`, `mean_j m_ij`, `sum_j a_ij m_ij` with
   `a_ij = softmax_j(w . m_ij)`, and the per-channel softmax aggregator
   `sum_j softmax_j(beta (.) m_ij) (.) m_ij` for any learned beta, over
   the N neighbors of i or over every satellite edge at the controller.
3. `sum_j a_ij M_ij` and `mean_j M_ij`.
4. `V W` for any c by c matrix W acting on the channel axis of V.
5. `||V||_c`, `<U V, W V>_c`, `<V, M>_c`, and `||M||_F`.
6. `LN(h)` for a layer norm over the scalar width.

Proof. (1) A function of an invariant scalar is an invariant scalar; the
gaussian grid is a fixed function of one column. (2) m_ij = phi_m(f_ij) is
a function of invariants, so it is invariant per edge; the neighbor set of
i is the same set before and after rotation because rho_R relabels
nothing, so any permutation-symmetric reduction over j (mean, max) is
invariant, and the softmax weights a_ij are functions of invariants, so
the weighted sum is too. (3) Each M_ij transforms by R and each a_ij is
fixed, so `sum_j a_ij R M_ij = R sum_j a_ij M_ij`; the mean is the case
a_ij = 1/N. (4) R acts on the 3 axis and W on the channel axis, so
`R (V W) = (R V) W` by associativity of the two matrix products on
different axes. (5) Each entry is an inner product or a norm of
co-rotating channel vectors, `<R a, R b> = <a, b>`. (6) LN reads only h,
which is invariant.
Consequence. Prop 3 survives the session 8 layer: every new map either
composes invariants or is a linear combination of equivariant vectors with
invariant coefficients.
Test: `spin_gnn/tests/test_message.py::test_message_equivariance`,
`spin_gnn/tests/test_update.py::test_update_equivariance`,
`spin_gnn/tests/test_equivariance.py::test_model_equivariant_fields_on_interior`,
`spin_gnn/tests/test_basis.py`. Status: verified.

Why these maps. The Task B label asks two questions the mean cannot answer
well: does any one of the N(N-1)/2 pairs align past TAU_U, and how many
satellites sit inside TAU_D. A mean over N neighbors carries one aligned
edge at weight 1/N, so the signal shrinks with N; max and attention carry
it at weight up to 1 (Corso et al. 2020, principal neighbourhood
aggregation; Xu et al. 2019 on what sum, mean, and max can distinguish).
The hard max passes gradient through one edge per channel, so it cannot
discover which edge feature to key on; the softmax aggregator with a
learned inverse temperature (Li et al. 2020, DeeperGCN) is the mean at
beta = 0 and the max as beta grows, with dense gradients throughout. The
controller also pools every satellite edge directly, so an edge signal
reaches the readout in one hop instead of two means. The raw edge
features are pooled by max and mean before any mixing: class 3 axes are
greedily packed with every pair below 0.9, while class 2 carries one
pair at exactly 1, so `max_ij rbf(alpha_ij)` at the center 1 reads 1.0
against at most exp(-1.18) = 0.31, a gap one linear weight can use from
the first step; a max over MLP outputs would first have to find the
column through the argmax edge. Measured on 100 constructed examples per
class (seed 123): the largest `abs(alpha_ij)` per example is at most 0.900
in class 3 and at least 0.960 in every other class, with class 2 at
exactly 1.000.
The gaussian grids let one linear layer read a threshold on d_ij or
alpha_ij as a bump (Schütt et al. 2017). The channel mixes and the vector
invariants are the gated equivariant block of PaiNN (Schütt, Unke, and
Gastegger 2021), which is what lets the vector state speak back to the
scalar state through more than one norm.

## 10. Complexity

Per layer: O(N^2 (d_e d_m + d_m c)) time, O(B N^2 d_m) memory, where d_e
is the packet width 14 + 2 d and d_m is the message width.
Measured wall-clock: 0.5277 seconds per step for the full configuration.

## 11. Limitations

phi is not a geometric spin; it is an invariant scalar with an exact
integrator. Reflections are untested, and the parity of omega is
undeclared by Prop 6. Equivariance at the walls is broken by design: the
projection commutes only with the cube symmetry group, per Prop 4.

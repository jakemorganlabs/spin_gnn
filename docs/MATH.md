# MATH: definitions, propositions, proofs, and their tests

Each proposition has a Statement, a Proof, and a Test line. Tests that
arrive in later sessions are listed with their future path and read
`pending`. Claims C1 through C11 live in CLAIMS.md.

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
Post-training step bound, seed 0: max_step 0.2328, within_bound False.

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

## 10. Complexity

Per layer: O(N^2 (d_e d_m + d_m c)) time, O(B N^2 d_m) memory, where d_e
is the packet width 14 + 2 d and d_m is the message width.
Measured wall-clock: 0.5277 seconds per step for the full configuration.

## 11. Limitations

phi is not a geometric spin; it is an invariant scalar with an exact
integrator. Reflections are untested, and the parity of omega is
undeclared by Prop 6. Equivariance at the walls is broken by design: the
projection commutes only with the cube symmetry group, per Prop 4.

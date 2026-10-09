# Cable dynamics as SOFA assembles it: remarks (2026-09-09, revised 2026-10-09)

Findings from `./scripts/run.sh cable_dynamics [--rom]`
(`cable_identification/dynamics_dump.py`), which reads every term of

    M(q) q'' + C(q,q') q' + f_int(q; EA,EI,GJ) = f_g + J_g(q)^T lambda

from the SOFA components of the truth cable (and of its POD-Galerkin ROM), writes
them to `artifacts/cable_dynamics/` (npz, png, full-text report, SOFA's own
`GlobalSystemMatrixExporter` files) and asserts the identities SOFA guarantees.
Details and the term-by-term table: [sofa_mor_pipeline.md §9](sofa_mor_pipeline.md).

**Revision 2026-10-09 (planar extensible model).** Sections 1-3 and 5-6 below now
describe the current model; section 4 is kept as the record of the plugin defect that
was fixed. What changed, and why (each item verified by `./scripts/run.sh
cable_fom_test` and the dump's identities; history in
[status §5.17](cable_ts_status_and_diagnosis.md)):

- strain state `Vec6d` `(kappa_x, kappa_y, kappa_z, eps_x, eps_y, eps_z)` per section
  (6 + 96 dofs), planar lock on torsion, bend_y and shear_z; Hooke law
  `-diag(GJ, EI, EI, EA, GA, GA) l (q - q0)` with finite EA = 5 kN, GA = 2 kN
  (nominal, uncalibrated); the strain components are increments over the straight
  rod, the plugin adds the unit axial stretch (`xi = (kappa, 1 + eps_x, eps_y, eps_z)`);
- `applyJ == d apply/dq` (`cosserat-patches/0002`: the tangent operator is built from
  the full twist): `J_g^T lambda` is the virtual work of the frames (dump ratio
  SOFA / finite differences 1.000000 on every bend_z row; `cable_fom_test`: Jacobian
  1.5e-10, `J^T lambda = -grad V_grasp` 2.7e-8);
- `M(q) = J^T diag(m, I_segment) J` with the rod-segment rotational inertia
  (`UniformMass.vertexMass = frame_rigid_mass(cfg)`; `totalMass` alone gave each
  frame a rotational inertia equal to its mass, 1.7e-3 instead of 5e-8 kg m^2);
- `C(q, q') q' = -B q'+`: Kelvin-Voigt damping as a real force
  (`DiagonalVelocityDampingForceField`, `B = -c_R diag(Sigma) l`, implicit). The
  solver's Rayleigh factors and the Hooke law's `rayleighStiffness` are 0;
  `rayleigh_mass_per_s = 0` (table friction neglected);
- convective inertia `-m_f (dJ_f/dt) q'` of the mapped masses (SOFA drops it): an
  optional explicit frame wrench (`convective_inertia`, `add_convective_inertia`),
  **off by default**. Measured along the excitation trajectories it is 0.2 % (median)
  to 3 % (max) of the elastic force; in a fast free oscillation 5 %, and its discrete
  work is 0.14 % of the Kelvin-Voigt dissipation. Explicit at h = 0.01 s it diverged
  once (softest vertex, cable snapping taut, t = 7.6 s of training trajectory 1);
  the same episode is stable at h = 0.005 s. Full-fidelity transients: enable it with
  `timestep_s: 0.005`;
- grasp: `RestShapeSpringsForceField` 1e5 N/m, 10 N m/rad on the grasped frame
  (no-slip assumption, configurable), target = gripper pose (+) the relative pose
  met at latch; no ramp, no reach clamp.

Time integration (measured, `cable_fom_test`): implicit Euler at h = 0.01 s removes
an extra 0.44x the Kelvin-Voigt dissipation in a free oscillation (numerical
dissipation), first-order convergence of the tip (1.40 mm -> 0.89 mm per halving of
h). Kept at 0.01 s: the live estimator's real-time budget; no accuracy failure of
the FOM gates required a smaller step.

## 1. How the terms are obtained

- `build_cable(..., expose_matrices=True)` puts a `MatrixLinearSystem` (solved) plus
  observer systems (`assembleMass` / `assembleDamping` / `assembleStiffness` one at a
  time, `applyMappedComponents=0` for the non-mapped stiffness) on the solver node,
  composed by `CompositeLinearSystem`; `CableHandles.system_matrices()` returns them
  as numpy/scipy through the shipped `Sofa.SofaLinearSystem` binding (`A()/b()/x()`).
- Everything is read, nothing re-derived: `q, q'` from the MechanicalObjects, `q''`
  from the solver solution, `lambda` from `FramesMO.force[tip]`, `J_g` by probing
  `applyJ` with unit velocities, `K_int` from the Hooke force field's own block.
- `SofaMatrix` (`GlobalSystemMatrixExporter`) IS shipped with the 25.12 binary
  (`/opt/sofa/plugins/SofaMatrix`); its text files match the binding to 17 digits.

## 2. What SOFA really solves (EulerImplicitSolver.cpp v25.12)

    A dq' = b,   A = (1 + h rM) M - h B - h (h + rK) K,   b = h (f + ((h + rK) K - rM M) q')

- `K = df/dq` (negative for a spring), `B = df/dv`: the Kelvin-Voigt force field is
  the only component with a damping matrix (`B = -c_R diag(Sigma) l` on the strains,
  zero on the base). Until 2026-10-09 `B == 0` and all damping was Rayleigh through
  the solver factors, which also damped the clamp and grasp springs.
- A force field's own `rayleighStiffness` (`rK_ff`, the Hooke law here) enters `A`
  only (`kFactorIncludingRayleighDamping`, RHS uses `B(0)`): it damps the velocity
  *increment*, `-rK_ff K_int dq'` (1e-5 N m at the dumped state), not a `rK_ff K q'`
  force. Earlier "Hooke damped with 2 rK" statements were wrong; corrected in §9.
- Rewritten, SOFA's step is exactly
  `M q'' + (rM M - B - rK K) q'+ - rK_ff K_int dq' = f(q) + h K q'+ ~ f(q+)`;
  the dump verifies this on the free dofs (residual 2.2e-15 on 2026-10-09).
- SOFA computes **no Coriolis/centrifugal term** for mapped masses: `M(q)` is
  re-assembled every step and `d(J q')/dt` at fixed `q'` is dropped. The scene can add
  it explicitly (see the revision note); by default `C(q,q') q'` is the Kelvin-Voigt
  term only.
- `MechanicalObject.derivX` is not `dq'`; `RestShapeSpringsForceField<Rigid>` returns
  zero torque below ~1e-7 rad (deadband).

## 3. The "finite element" form of the cable

The SOFA doc's FEM chapter (shape functions, `K = sum_e int B^T D B`) describes the
volume elements (`TetrahedronFEMForceField` ...). The cable is a Cosserat rod in
strain coordinates (SofaDefrost `Cosserat`, piecewise-constant strain):

- element = one of 16 sections, constant strain `(kappa_x, kappa_y, kappa_z, eps_x,
  eps_y, eps_z)`; dofs = 6 rigid base + 96 strains = 102 (48 free in the table setup).
- "shape functions" = `DiscreteCosseratMapping.apply`: SE(3) exponentials
  `g(s) = g_base prod exp(l_i xi_i^)`. This is the only nonlinearity.
- element stiffness = `BeamHookeLawForceField<Vec6d>`: `K_e = -diag(GJ, EI, EI, EA,
  GA, GA) l`, block-diagonal and **constant** (3.5e-4, 4.375e-4, 4.375e-4 N m^2 and
  218.75, 87.5, 87.5 N per section here); the axial and shear stiffnesses are 5-6
  decades above the bending ones, so the rod is nearly inextensible in bending
  motions and stretches by `F L / EA` under tension (1.4 mm at 10 N).
- no mass on the strains: `UniformMass` (0.07 kg, rod-segment inertia) on 41 mapped
  frames, projected `M(q) = J^T diag(m, I) J` -> dense, configuration dependent, SPD,
  cable mass on the base translation block.
- boundary conditions: `PartialFixedProjectiveConstraint` (planar; identity rows in
  `A` on 48 dofs), penalty springs 1e8 on the base and 1e5 N/m / 10 N m/rad at the
  grasped frame.
- time integration: one linearised implicit Euler step per `h = 0.01 s`, no Newton.

## 4. Finding (2026-09-09, fixed by `cosserat-patches/0002`): the pinned Cosserat
## plugin's Jacobian was not the derivative of its mapping

Pinned release f64e029 (`DiscreteCosseratMapping`): `applyJT == applyJ^T` (checked),
but `applyJ` is **not** `d(apply)/dq`. On the straight rod the tip moves by
`l (L - s_i)` per unit bend rate of section `i` (rotation about the section START
node) instead of `l (L - s_i - l/2)` (curvature distributed over the section):
the tangent operator `updateTangExpSE3` lacked the angular-linear coupling: it
built `ad_xi` from `(kappa, 0, 0, 0)` while the exponential used the unit axial
strain. At the dumped bent state `J_g^T lambda` differed from a consistent Jacobian
by 1.5-3 % at the base sections, 27-58 % near the tip, and flipped sign on the last
one. `M`, `J^T K_g J` and `J_g^T lambda` all used this `J`. Patch 0002 feeds the
full twist to the tangent operator (both Vec3 and Vec6 routes) and fixes the Vec6
specialisations (strain-rate index bug in `applyJ`, outputs copied by value and
linear-strain rows zeroed in `applyJT`, uninitialised `d_EIy/d_EIz` in the Hooke
law). Confirmed consequence: the grasp spring's generalized force was not the
gradient of its potential, so forces on mapped frames did net work over cycles;
with the consistent Jacobian the 600 s holds under bending and sustained tension of
[status doc §5.16](cable_ts_status_and_diagnosis.md) are stable at 1e5 N/m.

## 5. The POD-Galerkin form (`--rom`)

`q = q_0 + Phi a` through `ModelOrderReductionMapping` (`Vec1d -> Vec6d`, MOR patch
0004); `Phi` is W-orthonormal in the strain-energy metric `W = diag(Sigma) l`
(`Phi^T W Phi = I`, projection `a = Phi^T W (q - q_0)`); unknowns = base (6) + modal
coordinates `a` (r). SOFA assembles the reduced operators through the mapping chain
exactly as for the FOM (the virtual-work projection is `Phi^T (.) Phi`, no extra `W`):

    M_r = Phi^T J^T diag(m, I) J Phi,  K_int_r = Phi^T K_int Phi,  B_r = Phi^T B Phi,
    K_grasp_r = Phi^T J^T K_g J Phi,  f_int_r = -Phi^T K_int (q - q_0),
    J_r^T lambda = Phi^T J_g^T lambda

- Phi comes from the plugin's SVD of the energy-scaled `WriteState` strain snapshots
  `W^1/2 q` (singular values 13.4, 9.3, 3.8, 2.4, 2.2, 0.90, ...; zero after 48 = the
  locked rows), unscaled to `Phi = W^-1/2 Phi~`; torsion/bend_y/shear_z rows exactly
  zero, `q_0 = 0`.
- **Not dumped yet for the current model**: no basis is promoted
  ([sofa_mor_pipeline.md §3](sofa_mor_pipeline.md)). `check_rom` now asserts
  `Phi^T W Phi = I` and `B_r = Phi^T B Phi` besides the Galerkin identities. The
  2026-09-09 ROM dump (three-strain plant: Galerkin identities to 1e-8, reduced
  residual 6e-15 at r = 16 and 1.6e-15 at r = 3) is what `cable_dynamics_rom.*`,
  `r3/` and `sofa_export_rom/` in `artifacts/cable_dynamics/` still contain.
- No hyper-reduction (ECSW): the Hooke law and the damping are 96 diagonal entries,
  nothing to sample; the ROM still evaluates the Cosserat mapping on all sections, so
  it is slower than the FOM from r ~ 9 on (measured 1.20x faster at r = 5, 0.67x at
  r = 18).
- With the MOR mapping the Hooke law is itself a mapped component, so the
  "non-mapped" observer isolates only the clamp; `system_matrices()` returns the
  mapped block as one pre-multiplied term for the ROM.

## 6. Shipped-binary pitfalls met on the way

- `import Sofa.SofaLinearSystem` **before** `addObject`, or the `MatrixLinearSystem`
  handle has no `A()/b()/x()`.
- `CompositeLinearSystem` (25.12) never raises its own `factorizationInvalidation`:
  the solver never factorises (dq' = 0, cable frozen) unless that Data is linked to
  the solved system's.
- Observer systems must be `template="FullMatrix"`: an unsolved CRS matrix is never
  compressed and reads back as zeros. `FullMatrix.A()` is a view on SOFA memory:
  copy before the next step or `unload`.
- Compare ROM and FOM at the state at the START of the step (the strain MO after
  the step already holds `q+`).

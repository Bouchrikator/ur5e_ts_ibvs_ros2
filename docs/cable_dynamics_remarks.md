# Cable dynamics as SOFA assembles it: remarks (2026-09-09)

Findings from `./scripts/run.sh cable_dynamics [--rom]`
(`cable_identification/dynamics_dump.py`), which reads every term of

    M(q) q'' + C(q,q') q' + f_int(q; EA,EI,GJ) = f_g + J_g(q)^T lambda

from the SOFA components of the truth cable (and of its POD-Galerkin ROM), writes
them to `artifacts/cable_dynamics/` (npz, png, full-text report, SOFA's own
`GlobalSystemMatrixExporter` files) and asserts the identities SOFA guarantees.
Details and the term-by-term table: [sofa_mor_pipeline.md §9](sofa_mor_pipeline.md).

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

- `K = df/dq` (negative for a spring). `B == 0`: no cable component implements
  `buildDampingMatrix`; all damping is Rayleigh through the factors.
- A force field's own `rayleighStiffness` (`rK_ff`, the Hooke law here) enters `A`
  only (`kFactorIncludingRayleighDamping`, RHS uses `B(0)`): it damps the velocity
  *increment*, `-rK_ff K_int dq'` (1e-5 N m at the dumped state), not a `rK_ff K q'`
  force. Earlier "Hooke damped with 2 rK" statements were wrong; corrected in §9.
- Rewritten, SOFA's step is exactly
  `M q'' + (rM M - rK K) q'+ - rK_ff K_int dq' = f(q) + h K q'+ ~ f(q+)`;
  the dump verifies this to 6e-15 on the free dofs.
- SOFA computes **no Coriolis/centrifugal term** for mapped masses: `M(q) = J^T diag(m) J`
  is re-assembled every step and `d(J^T M J)/dt` is dropped. `C(q,q') q'` in the
  equation above is only the Rayleigh term.
- `MechanicalObject.derivX` is not `dq'`; `RestShapeSpringsForceField<Rigid>` returns
  zero torque below ~1e-7 rad (deadband).

## 3. The "finite element" form of the cable

The SOFA doc's FEM chapter (shape functions, `K = sum_e int B^T D B`) describes the
volume elements (`TetrahedronFEMForceField` ...). The cable is a Cosserat rod in
strain coordinates (SofaDefrost `Cosserat`, piecewise-constant strain):

- element = one of 16 sections, constant strain `(kappa_x, kappa_y, kappa_z)`;
  dofs = 6 rigid base + 48 strains = 54.
- "shape functions" = `DiscreteCosseratMapping.apply`: SE(3) exponentials
  `g(s) = g_base prod exp(l_i xi_i^)`. This is the only nonlinearity.
- element stiffness = `BeamHookeLawForceField`: `K_e = -diag(GI, EI, EI) l`,
  block-diagonal and **constant** (3.5e-4 / 4.375e-4 N m^2 per section here).
- no mass on the strains: `UniformMass` (0.07 kg) on 41 mapped frames, projected
  `M(q) = J^T diag(m) J` -> dense, configuration dependent, SPD, cable mass on the
  base translation block.
- boundary conditions: `PartialFixedProjectiveConstraint` (planar; identity rows in
  `A` on 32 dofs, `A` has 400 nnz vs 1580 for the observers), penalty springs 1e8 on
  the base and 2e3 at the tip. Spring stiffnesses dominate `K` by 4 to 11 decades
  over the Hooke law.
- time integration: one linearised implicit Euler step per `h = 0.01 s`, no Newton.

## 4. Finding: the Cosserat plugin's Jacobian is not the derivative of its mapping

Pinned release f64e029 (`DiscreteCosseratMapping`): `applyJT == applyJ^T` (checked),
but `applyJ` is **not** `d(apply)/dq`. On the straight rod the tip moves by
`l (L - s_i)` per unit bend rate of section `i` (rotation about the section START
node) instead of `l (L - s_i - l/2)` (curvature distributed over the section):
the tangent operator `updateTangExpSE3` lacks the angular-linear coupling. At the
dumped bent state `J_g^T lambda` differs from a consistent Jacobian by 1.5-3 % at the
base sections, 27-58 % near the tip, and flips sign on the last one. `M`,
`J^T K_g J` and `J_g^T lambda` all use this `J`, so the equation is exact for SOFA's
own `J` but not the virtual work of the frames' motion. Recorded, not changed
(it is the plant everything was identified on). Hypothesis, not a result: candidate
explanation for the energy pumped into the rod by forces on mapped frames
([status doc §5.16](cable_ts_status_and_diagnosis.md)).

## 5. The POD-Galerkin form (`--rom`)

`kappa = kappa_0 + Phi a` through `ModelOrderReductionMapping`; unknowns = base (6)
+ modal coordinates `a` (r). SOFA assembles the reduced operators through the mapping
chain exactly as for the FOM:

    M_r = Phi^T J^T diag(m) J Phi,  K_int_r = Phi^T K_int Phi,  K_grasp_r = Phi^T J^T K_g J Phi,
    f_int_r = -Phi^T K_int (kappa - kappa_0),  J_r^T lambda = Phi^T J_g^T lambda

- Phi comes from the plugin's SVD of `WriteState` strain snapshots (singular values
  874, 685, 204, 110, 78, 54, 27, 17, ...; zero after 16 = the 32 planar-locked rows);
  orthonormal columns, torsion/bend_y rows exactly zero, `kappa_0 = 0`.
- Verified Galerkin property: SOFA's assembled `M_r` and `K_r_term` equal
  `Phi^T (FOM placed at kappa_0 + Phi a) Phi` to 1e-8 (the 5-decimal precision of
  the mode file; 3e-9 / 1.5e-13 measured), `kappa == kappa_0 + Phi a` exactly, the
  reduced equation balances to 6e-15 (r = 16) and 1.6e-15 (r = 3). The reduction is
  an intrusive projection of the full operators, not a fit.
- No hyper-reduction (ECSW): the Hooke law is 48 diagonal entries, nothing to sample.
- r = 16 is the full planar rank -> a change of basis, no speed-up (§3 of the
  pipeline doc). r = 3 (`candidates/tol_0.2`, 98.27 % energy) gives a 3 x 3 modal
  mass and 3 x 3 `Phi^T K_int Phi`; dumped to `artifacts/cable_dynamics/r3/`.
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

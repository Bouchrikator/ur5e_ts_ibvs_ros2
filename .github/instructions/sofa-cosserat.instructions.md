---
description: "Use when editing SOFA Cosserat cable scenes, grasp coupling, cable YAML configs, the Optimus estimator, the SOFA<->ROS 2 adapter, or Cosserat/Optimus installers, source and patches. Covers SOFA v25.12 API pitfalls, mapped-state constraints/forces, planar clamp, grasp spring, Optimus ROUKF wiring, plugin rebuilds and runSofa GUI quirks."
applyTo: "src/cable_identification/**, src/sofa_ros2_adapter/**, src/cable_ts_control/cable_ts_control/sofa_*.py, src/cable_ts_control/cable_ts_control/cable_modal_observer_node.py, src/cable_ts_control/config/cable_mor.yaml, scripts/install_cosserat.sh, scripts/install_optimus.sh, scripts/install_model_order_reduction.sh, third_party/cosserat-patches/**, third_party/Optimus/**, third_party/model-order-reduction-patches/**"
---
# SOFA v25.12 + Cosserat cable scene rules

Every rule below was a real bug (docs §5.1, §5.2, §5.16 and the 2026-08 fixes).
Check [docs/cable_ts_status_and_diagnosis.md](../../docs/cable_ts_status_and_diagnosis.md)
§5 before changing anything in the scene; do not reintroduce a listed bug.

## Scene construction ([cosserat_model.py](../../src/cable_identification/cable_identification/cosserat_model.py))
- Root needs exactly one explicit animation loop (v25.12 errors in `animate()` without
  one): `FreeMotionAnimationLoop` + `BlockGaussSeidelConstraintSolver` for the Lagrange end
  attachment (`planar` + `grasp_tip`), `DefaultAnimationLoop` for force-only legacy scenes.
  `prepare_root(animation_loop="auto")` picks it; `None` = the caller adds its own. Never both.
- v25.12 renamed `GenericConstraintSolver` to `BlockGaussSeidelConstraintSolver`.
- The strain MO is `Vec6d`, section-major `(kappa_x, kappa_y, kappa_z, eps_x, eps_y,
  eps_z)` as increments over the straight rod (the plugin adds the unit axial stretch
  itself: `xi = (kappa, 1 + eps_x, eps_y, eps_z)`). `BeamHookeLawForceField` template
  `Vec6d` with `useInertiaParams` (EI, GI, EA, GA); the pinned release's Vec6 route is
  only usable with `cosserat-patches/0002` (uninitialised `d_EIy/d_EIz`, `applyJ` index
  bug, `applyJT` copying its outputs and zeroing the linear-strain rows).
- Projective constraints act on the **independent** Cosserat strain MO, never on the mapped
  `FramesMO`: a `PartialFixedProjectiveConstraint` on mapped dofs is a silent no-op
  ("only main mechanical states have an associated submatrix"). Planar =
  `fixedDirections [1,1,0,0,0,1]` on the strain MO (torsion, bend_y, shear_z locked;
  bend_z, extension, in-plane shear free); the ROM gets planarity from the basis.
  Lagrange rows may originate on mapped frames when their propagation to the independent
  dofs is implemented and tested: `PlanarAttachmentConstraint` on `FramesMO` reaches the
  strains/modes through the mappings' `applyJT(MatrixDeriv)` (`cable_fom_test --checks
  grasp`: actual `H_q` and `H_a` vs FD, `H_a == H_q Phi`).
- `DiscreteCosseratMapping.applyJ` IS the derivative of `apply()` since
  `cosserat-patches/0002` (the tangent operator is built from the full twist, rest axial
  strain 1); `applyJT == applyJ^T`. Checked by `cable_fom_test` (FD 1e-10, virtual work
  3e-8) and `cable_dynamics`. The old energy pumping of forces on mapped frames (status
  doc 5.16) came from that inconsistent Jacobian: with it fixed, 600 s holds under
  bending and sustained tension are stable with a 1e5 N/m grasp spring.
- Since `cosserat-patches/0003` the exponential and tangent coefficients are Taylor series
  in `z = l |kappa|` for `|z| < 2`: the closed forms lost all digits near zero curvature
  (2 % Jacobian error at `|kappa| ~ 1e-9`, status doc 5.18). `cable_fom_test --checks
  kinematics restore` covers zero/near-zero/switch curvatures against an independent `expm`
  chain and central differences; keep FD probes on `save_state` + `finally: restore_state`.
- Frame mass: `UniformMass(vertexMass=frame_rigid_mass(cfg))`, never `totalMass` alone
  (that leaves `inertiaMatrix` at identity, a rotational inertia equal to the mass:
  1.7e-3 instead of 5e-8 kg m^2 per frame). Damping: `DiagonalVelocityDampingForceField`
  on the strains (`rayleigh_stiffness_s * diag(Sigma) l`, Kelvin-Voigt), the solver's
  `rayleighStiffness` and the Hooke law's are 0; `rayleigh_mass_per_s` is table/air
  friction and is 0. The convective inertia `-m (dJ/dt) q'` is an optional explicit
  frame wrench (`convective_inertia`, `add_convective_inertia`), off by default: at
  h = 0.01 s it diverged when the softest cable snapped taut (stable at 0.005 s);
  enable it only with `timestep_s <= 0.005`. The dump and `cable_fom_test` account for it.
- Planar table scene (`planar` + `grasp_tip`): the base is the fixed fixture
  (`FixedProjectiveConstraint`, no anchor spring; it never projects positions, so only
  `GraspCoupling.on_fixture`/`reset_fixture` write the pose) and the end s = L is held by
  the three bilateral rows `[x, y, yaw]` of `PlanarAttachmentConstraint`
  (`cosserat-patches/0004`), solved with `GenericConstraintCorrection` linked to the
  cable's solvers (`LinearSolverConstraintCorrection` has no Vec6 template and covers one
  MO only). Its multiplier is an impulse: reaction = `lambda / h` (checked at h and h/2).
  History and regression condition: a six-row `BilateralLagrangianConstraint` through
  `DiscreteCosseratMapping` with a spring-held base dragged the base (400 mm error);
  `cable_fom_test --checks fixture grasp` must keep base drift <= 1e-9 m / rad.
- Legacy branch only (non-planar `grasp_tip`): the penalty grip `RestShapeSpringsForceField`
  (`grasp_stiffness` 1e5 N/m, angular 10 N m/rad) on the grasped frame, moved with
  `set_grasp_point` at latch. It is not an exact attachment.
- Never call `reinit()` on a component before `Simulation.init` has resolved its
  links: runSofa segfaults (exit code -11), it does not raise. Guard as
  `set_grasp_spring_enabled` does (`findLink("mstate").getLinkedBase() is not None`).
  The GUI scene builds the coupling pre-init, the headless nodes post-init, so a
  change must be exercised on both paths (`cable_forward_test` check 7 is pre-init).
- Keep the marker indices, `L`, substep/`control_dt` ratio and `marker_s_over_l`
  consistent with the basis/model YAML; the offline pipeline reads them from there.
- Assembled matrices (`build_cable(..., expose_matrices=True)`, `CableHandles.system_matrices()`,
  `./scripts/run.sh cable_dynamics`, doc section 9): `import Sofa.SofaLinearSystem` **before**
  `addObject` or the `MatrixLinearSystem` handle has no `A()/b()/x()`. `CompositeLinearSystem`
  in 25.12 never raises its own `factorizationInvalidation`, so the solver never factorises
  (dv = 0, cable frozen) unless that Data is linked to the solved system's. Observer systems
  must be `template="FullMatrix"` (an unsolved CRS matrix is never compressed and reads back
  as zeros) and `FullMatrix.A()` is a view on SOFA memory: `np.array(..., copy=True)` before
  the next step or `unload`. `applyMappedComponents=False` separates the Hooke law and the
  clamp from the projected grasp spring. The documented `GlobalSystemMatrixExporter` IS
  shipped (plugin `SofaMatrix` under `/opt/sofa/plugins`, not `/opt/sofa/include`).
  A force field's own `rayleighStiffness` enters `A` only (`-h (h + rK + rK_ff)`), never the
  RHS; the Kelvin-Voigt damping is a real force (`B = df/dv` in the damping observer).

## Grasp coupling ([coupling.py](../../src/cable_identification/cable_identification/coupling.py))
- Planar table scene: latch only within `attach_distance_m` (0.010 m, 3D) of the end
  s = L (mapping refreshed first); `T_offset = T_g^-1 T_end` is captured once and the
  target is `T_g T_offset` with twist `v_g + w_g x (R_g p_offset), w_g` until release.
  Commands whose composed target leaves the plane (> `attach_plane_tolerance_m`) or tilts
  the section normal (> `attach_tilt_rad`; both 1e-8, numerical tolerances of the ideal
  planar benchmark; `attach_distance_m` is the latch distance only) raise
  `IncompatibleGraspCommand` (the live nodes log it, the
  target is held); never project them silently. Release only removes the rows. The
  fixture is placed once per run; a moved fixture needs `reset_fixture`.
  `checkpoint()`/`restore()` is the rollout replay state (mechanics, time, attachment).
- Legacy spring grip: the spring is disabled while DETACHED. On the latch edge the frame
  nearest to the gripper becomes the grasped frame (`set_grasp_point`; the cable beyond it
  is free) and the relative pose gripper -> frame is captured, so the target
  `gripper (+) offset` starts exactly at rest: no ramp, no reach clamp. Pulling past
  the rest length stretches the rod (EA), it is not slip. On clamp use the analytic
  straight tip (fixture + `[L,0,0]`), not the stale mapped frame.
- Excitation: declared range `radial_min`..`radial_max` in rest lengths (0.92 bends,
  1.002 pulls taut: 1.4 mm = 10 N at EA = 5 kN) and `yaw_limit`; pushing far inward
  buckles the rod (unfittable bifurcation).

## Adapter and GUI scene
- Exactly one physics owner: `cable_sofa_node` with `role:=truth`, or the runSofa
  scene [cable_scene.py](../../src/cable_identification/cable_identification/cable_scene.py)
  when `sofa_gui:=true` (the node is then disabled by the launch). Both publish the
  same `/cable/truth/{frames,markers}` + `/cable/grasp_state` contract; the node also
  publishes `/cable/truth/solver_stats` = `[compute_ms, backlog_steps, stepped]`, and a
  rising backlog means CPU starvation (the estimator then rejects most
  observations): fix the load, never the gate.
- runSofa evaluates `createScene` twice: guard `rclpy` init with `rclpy.ok()` and
  use a unique node name.
- Never add `/opt/sofa/lib` to a global `LD_LIBRARY_PATH` (Qt 5.12 vs 5.15); use the
  `runSofa` wrapper or scope it per process. `SOFA_ROOT=/opt/sofa`, SofaPython3 on
  `PYTHONPATH`, as done in [qs_live_test.sh](../../scripts/cable/qs_live_test.sh).
- Cosserat lives in the container overlay; `scripts/install_cosserat.sh` restores it
  (idempotent) and `run.sh in_live` calls it before every cable command.
  `scripts/install_optimus.sh` then replaces it with the source build carrying the
  EI/GI internal-data patch and installs Optimus; both are no-ops when present.
- The estimator identifies in log space through Optimus (`transformParams=exponential`);
  the legacy `LogParameterUKF` rules still hold for that library: clamp `|q| <= 50`
  and gate on the innovation (`exp()` underflow silently zeroes the stiffness).

## Optimus estimator ([optimus_scene.py](../../src/cable_identification/cable_identification/optimus_scene.py), [docs/optimus_port.md](../../docs/optimus_port.md))
- `OptimParams.stdev` is NOT a physical std: for a coefficient of variation `c`
  pass `exp(sqrt(log(1+c^2)))` (`optimus_stdev`), 1.341194157207 for c = 0.30.
- Every sigma point must see the same boundary pose in a filter step (the boundary
  controller is hit p+1 times per step); never read "the current time" inside it.
- ROUKF needs positions in the state (`estimatePosition=True`); the wrapper needs the
  explicit `mstate` path (several MOs live under the cable node).
- No valid observation for a step = prediction only (`set_observation(..., valid=False)`),
  never a zero-filled observation. Missing markers = NaN rows: they get zero weight in
  R^-1 (`observationVariances`), the correction uses the observed ones (dof = 3 x observed).
- The innovation gate is the filter's NIS test after prediction (`innovationGateSigma`,
  node `innovation_gate_sigma` = 3); do not add a Python RMS gate before the step, it
  compares the observation with the state one dt earlier.
- The estimator node owns model time: step to each observation's stamp, TF at the step
  time with no fallback to "latest", drop stale observations, never jump `_sim_time`
  without stepping. Time-consistency bugs here bias EI (9 % measured).
- GJ with `planar: true` is refused (kappa_x = 0, zero information); with the
  physical frame inertia GJ is not observable from centerline markers under a base
  roll either (gate G is a `[LIMIT]`). EI in the table setup is NOT observable from
  marker positions with the no-slip grasp (0.2 sigma_obs, gate I reports `[LIMIT]`):
  identify EI hanging under gravity (gate F) or add the wrist wrench; never soften
  the grasp to make it observable (optimus_port.md §7).
- The plain plant applies a base pose one solve late; when comparing truth and
  estimator in one process call `CableHandles.refresh_mapping()` after `set_base_pose`.
  Do not change the plant's default (certified artifacts depend on it).
- After editing the wrapper/patch/scene run `./scripts/run.sh optimus_smoke_test`,
  `cable_optimus_test` (= `ros2 run cable_identification optimus_recovery_test`) and
  `cable_optimus_pipeline` (= `optimus_pipeline_test`), in that order; rebuild
  `cable_identification` + `sofa_ros2_adapter` first (ament_python copies), and
  `scripts/install_optimus.sh --force optimus|cosserat` after a `third_party/` edit.

## Validating a scene change
Headless A/B scripts that step the real `cable_identification` code
(`PYTHONPATH=/ros2_ws/src/cable_identification`), long holds (>= 600 s), with
chain length and max |strain| as divergence metrics; then
`./scripts/run.sh cable_plugin_test`, `cable_forward_test` and `cable_fom_test`
(`*_PASSED` lines; the last checks traction, bending, Jacobian/virtual work, grasp,
energy and convergence against closed forms), and `cable_dynamics` for the
assembled-matrix identities.

## Strain-POD ROM (ModelOrderReduction), [docs/sofa_mor_pipeline.md](../../docs/sofa_mor_pipeline.md)
- `build_cable(..., reduction=ReductionSpec(modes, metadata, r))` inserts
  `modalCoordinateMO` (Vec1d, independent) -> `ModelOrderReductionMapping`
  (`Vec1d->Vec6d`, MOR patch 0004) -> `cosseratCoordinateMO` (now MAPPED: no projective
  constraint on it, `save_state`/`restore_state` act on the modal state,
  `refresh_mapping` re-applies MOR then Cosserat). `reduction=None` is the unchanged FOM.
- The POD lives in the strain-energy metric `W = diag(Sigma) l` (`strain_weights`): the
  official POD runs on `W^1/2 q`, the stored basis is `Phi = W^-1/2 Phi~` with
  `Phi^T W Phi = I`, and every projection is `a = Phi^T W (q - q0)` (`project_strain`),
  never `Phi^T q`. Order selection uses the validation trajectories only; the test
  trajectory is evaluated once with the frozen order (`validate_cosserat_rom`).
- `WriteState` records the strain MO on SOFA's continuous clock: pass
  `time=[root.time.value]`, never reset `root.time` (child contexts lag one step
  and the first record is duplicated); labels are indented (`  X=`).
- The only POD is the plugin's `readStateFilesAndComputeModes` on the (energy-scaled)
  training `WriteState` files; never fit a basis on markers for `a`. Its modes file has
  5 decimals: the r = 48 full-rank basis still leaves a 2.5e-5 projection residual.
- Fits in `a` must use the mass metric from the scene mapping
  (`sofa_modal_ts_identification.mass_metric`): the PSD projection in raw strain
  coordinates destroys the model. Rollout metrics must be windowed (index 0 is
  a hold).
- LMIs above ~20 states: `solve_cable_ts_lmi --backend sparse` (SCS); cvxpy and
  Clarabel are OOM-killed. The verifier is the gate, not the solver status.
- `cable_sofa_*` and `cable_fom_test` (its restore and grasp checks build the full-space
  diagnostic ROM) install the pinned MOR plugin through `in_live`.

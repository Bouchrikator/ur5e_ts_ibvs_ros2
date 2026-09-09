---
description: "Use when editing SOFA Cosserat cable scenes, grasp coupling, cable YAML configs, the Optimus estimator, the SOFA<->ROS 2 adapter, or Cosserat/Optimus installers, source and patches. Covers SOFA v25.12 API pitfalls, mapped-state constraints/forces, planar clamp, grasp spring, Optimus ROUKF wiring, plugin rebuilds and runSofa GUI quirks."
applyTo: "src/cable_identification/**, src/sofa_ros2_adapter/**, src/cable_ts_control/cable_ts_control/sofa_*.py, src/cable_ts_control/cable_ts_control/cable_modal_observer_node.py, src/cable_ts_control/config/cable_mor.yaml, scripts/install_cosserat.sh, scripts/install_optimus.sh, scripts/install_model_order_reduction.sh, third_party/cosserat-patches/**, third_party/Optimus/**, third_party/model-order-reduction-patches/**"
---
# SOFA v25.12 + Cosserat cable scene rules

Every rule below was a real bug (docs §5.1, §5.2, §5.16 and the 2026-08 fixes).
Check [docs/cable_ts_status_and_diagnosis.md](../../docs/cable_ts_status_and_diagnosis.md)
§5 before changing anything in the scene; do not reintroduce a listed bug.

## Scene construction ([cosserat_model.py](../../src/cable_identification/cable_identification/cosserat_model.py))
- Root needs an explicit `DefaultAnimationLoop` (v25.12 errors in `animate()` without it).
- v25.12 renamed `GenericConstraintSolver` to `BlockGaussSeidelConstraintSolver`.
- Constraints and forces act on the **independent** Cosserat strain MO, never on the
  mapped `FramesMO`: a `PartialFixedProjectiveConstraint` on mapped dofs is a silent
  no-op ("only main mechanical states have an associated submatrix"), and a force on
  mapped dofs has no geometric stiffness in the implicit solver, so any sustained
  load pumps energy until the rod coils. Planar = `fixedDirections [1,1,0]` on strain.
- Tip grasp = `RestShapeSpringsForceField` (`grasp_stiffness` 2e3, angular 0.05) to
  an external target MO. `BilateralLagrangianConstraint` through
  `DiscreteCosseratMapping` dragged the base (400 mm error): do not go back to it.
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
  RHS. `DiscreteCosseratMapping.applyJ/applyJT` are mutually consistent but are NOT the
  derivative of `apply()` (lumped-node lever `l (L - s_i)`): do not build a check that
  assumes `J^T lambda` equals the virtual work of the frames.

## Grasp coupling ([coupling.py](../../src/cable_identification/cable_identification/coupling.py))
- The spring is disabled while DETACHED and re-enabled on the latch edge with its
  target synced to the current tip; on clamp use the analytic straight tip
  (fixture + `[L,0,0]`), not the stale mapped frame, or the cable yanks/curls.
- `_clamp_reachable` projects the anchor to 99 % of `L` around the fixture. `q*=0`
  (straight) sits on that boundary, so over-extension is the operating point.
- Excitation must stay in the annulus 0.92–0.995 `L`: pushing inward buckles the
  inextensible rod (unfittable bifurcation).

## Adapter and GUI scene
- Exactly one physics owner: `cable_sofa_node` with `role:=truth`, or the runSofa
  scene [cable_scene.py](../../src/cable_identification/cable_identification/cable_scene.py)
  when `sofa_gui:=true` (the node is then disabled by the launch). Both publish the
  same `/cable/truth/{frames,markers}` + `/cable/grasp_state` contract; the node also
  publishes `/cable/truth/solver_stats` = `[compute_ms, dropped_steps, stepped]`, and a
  rising `dropped_steps` means CPU starvation (the estimator then rejects most
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
- GJ with `planar: true` is refused (kappa_x = 0, zero information). EI in the table
  setup is identified relative to the tip spring `k_theta L/EI = 3.5`; for hardware
  identify EI hanging under gravity or calibrate the clamp (optimus_port.md §7).
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
`./scripts/run.sh cable_plugin_test` and `cable_forward_test` (`*_PASSED` lines).

## Strain-POD ROM (ModelOrderReduction), [docs/sofa_mor_pipeline.md](../../docs/sofa_mor_pipeline.md)
- `build_cable(..., reduction=ReductionSpec(modes, metadata, r))` inserts
  `modalCoordinateMO` (Vec1d, independent) -> `ModelOrderReductionMapping` ->
  `cosseratCoordinateMO` (now MAPPED: no projective constraint on it, `save_state`
  /`restore_state` act on the modal state, `refresh_mapping` re-applies MOR then
  Cosserat). `reduction=None` is the unchanged FOM.
- `WriteState` records the strain MO on SOFA's continuous clock: pass
  `time=[root.time.value]`, never reset `root.time` (child contexts lag one step
  and the first record is duplicated); labels are indented (`  X=`).
- The only POD is the plugin's `readStateFilesAndComputeModes` on the training
  `WriteState` files; never fit a basis on markers for `a`.
- Fits in `a` must use the mass metric from the scene mapping
  (`sofa_modal_ts_identification.mass_metric`): the PSD projection in raw strain
  coordinates destroys the model. Rollout metrics must be windowed (index 0 is
  a hold).
- LMIs above ~20 states: `solve_cable_ts_lmi --backend sparse` (SCS); cvxpy and
  Clarabel are OOM-killed. The verifier is the gate, not the solver status.
- `cable_sofa_*` commands install the pinned MOR plugin through `in_live`; the
  FOM commands do not need it.

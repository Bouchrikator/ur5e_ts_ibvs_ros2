---
description: "Use when editing the SOFA Cosserat cable scene, grasp coupling, cable YAML configs, the Optimus estimator scene or the SOFA<->ROS 2 adapter: cosserat_model.py, optimus_scene.py, coupling.py, cable_scene.py, cable_sofa_node.py, cable_estimator_node.py, cable_truth/estimator yaml. Covers SOFA v25.12 API pitfalls, mapped-state constraints/forces, planar clamp, grasp spring, Optimus ROUKF wiring, runSofa GUI quirks."
applyTo: "src/cable_identification/**, src/sofa_ros2_adapter/**"
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
- Keep the marker indices, `L`, substep/`control_dt` ratio and `marker_s_over_l`
  consistent with the basis/model YAML; the offline pipeline reads them from there.

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
  same `/cable/truth/{frames,markers}` + `/cable/grasp_state` contract.
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
  never a zero-filled observation.
- The plain plant applies a base pose one solve late; when comparing truth and
  estimator in one process call `CableHandles.refresh_mapping()` after `set_base_pose`.
  Do not change the plant's default (certified artifacts depend on it).
- After editing the wrapper/patch/scene run `optimus_smoke_test`, `cable_optimus_test`,
  `cable_optimus_pipeline`, in that order; rebuild `cable_identification` +
  `sofa_ros2_adapter` first (ament_python copies).

## Validating a scene change
Headless A/B scripts that step the real `cable_identification` code
(`PYTHONPATH=/ros2_ws/src/cable_identification`), long holds (>= 600 s), with
chain length and max |strain| as divergence metrics; then
`./scripts/run.sh cable_plugin_test` and `cable_forward_test` (`*_PASSED` lines).

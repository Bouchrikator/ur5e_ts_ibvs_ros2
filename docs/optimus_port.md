# Optimus on SOFA v25.12 — port record

Status: **installed and gated**. Optimus' reduced-order UKF now runs inside the
existing `cable_estimator_node`; the in-process `LogParameterUKF` is no longer
used by the node (the module and its tests stay as a library). Written against
the portage protocol `optimus_sofa25_portage.md` (Optimus 1.1 → SOFA 25.12,
minimal core, log-space parameters, gates A–I).

## 1. Pinned versions

| Component | Version | Where |
|---|---|---|
| SOFA | v25.12.00 binary (`/opt/sofa`, SofaPython3) | [docker/Dockerfile](../docker/Dockerfile) |
| Optimus | upstream `sofa-framework/Optimus` @ `cd41bf17d1ae451aa7208ed0e2219d0702847bbd`, minimal core, vendored and modified | [third_party/Optimus](../third_party/Optimus) |
| Cosserat | `SofaDefrost/Cosserat` @ `f64e029de61d320700bcb5f0ececc4efc0d8b1d9` (= the commit of the shipped `release-v25.12` binary), rebuilt from source with one patch | [third_party/cosserat-patches](../third_party/cosserat-patches), [scripts/install_optimus.sh](../scripts/install_optimus.sh) |
| Toolchain | Ubuntu 24.04, gcc 13.3, cmake 3.28, make -j2 (7 GB host) | container |

One code path builds both plugins: `scripts/install_optimus.sh` runs in the
Dockerfile stage (`WITH_OPTIMUS=true`, default) and from `./scripts/run.sh`
(`in_live` calls it before every cable command; `./scripts/run.sh optimus`
runs it alone). It is idempotent: markers are `Cosserat/include/.../BeamHookeLawForceField.h`
declaring `doUpdateInternal` and `Optimus/lib/libOptimus.so`. The Cosserat
source is fetched as a commit-pinned tarball (git-over-https from the container
is flaky); GitHub does not publish a stable checksum for archive tarballs, so
the SHA is pinned and the archive is not checksummed.

Build dirs (`/opt/build/{cosserat-src,cosserat,optimus}`) are kept in a live
container for incremental rebuilds (`--force`) and deleted in the image.

## 2. What was ported (third_party/Optimus)

Minimal core only, as the protocol prescribes: `FilteringAnimationLoop`,
`ROUKFilter`, `StochasticStateWrapper`(+Base), `OptimParams`,
`MappedStateObservationManager`(+`ObservationManagerBase`),
`SimulatedStateObservationSource`(+`ObservationSource`),
`PreStochasticWrapper`, `FilterEvents`, `TimeProfiling`, `StochasticFilterBase`.
Not ported: UKF classic/ETKF/other filters, Verdandi bindings, Python helper
scenes, image/optical observation managers, tests.

Migration actually applied (SOFA 21.12 → 25.12):

| 21.12 | 25.12 | Notes |
|---|---|---|
| `SofaBaseMechanics`, `SofaSimulationCommon`, monolithic targets | `Sofa.Core/.DefaultType/.Helper/.LinearAlgebra`, `Sofa.Simulation.Core/.Common`, `Sofa.Component.StateContainer`, `.Constraint.Projective`, `.Constraint.Lagrangian.Solver`, `.Topology.Container.Dynamic` | modular CMake, `sofa_create_package_with_targets`, `OptimusConfig.cmake.in` |
| `initExternalModule` + static `SOFA_DECL_CLASS`/`RegisterObject` | `registerObjects(ObjectFactory*)` aggregating one `register*` per component | `optimusConfig.h` must `#define SOFA_TARGET Optimus`, otherwise `RequiredPlugin` reports "No component registered" |
| `templateName()` overrides | `static GetCustomTemplateName()` (`getTemplateName` is final) | Vector types default the template |
| `core::VecCoordId::position()`, `VecDerivId::dx()`, ... | `core::vec_id::write_access::position/velocity/dx/dforce/freePosition/freeVelocity`, `read_access::...` | |
| `ConstraintParams::POS/VEL/POS_AND_VEL` | `core::ConstraintOrder::POS/VEL/POS_AND_VEL` | |
| headers `sofa/simulation/*Visitor.h` | `sofa/simulation/mechanicalvisitor/*` | |
| `SOFA_OPTIMUSPLUGIN_API` from `optimusConfig.h.in` | hand-written `optimusConfig.h` (module name/version, `SOFA_STOCHASTIC_API`) | |

Behavioural changes (not pure API renames — each is exercised by a gate):

- `StochasticStateWrapper`: `mstate` data (explicit path to the wrapped
  `MechanicalObject`) — the Cosserat cable has several mechanical states below
  the wrapper node and the upstream "first one found" rule picked the wrong one;
  `getMappedPosFromFilterVector` (re-apply a sigma point's positions and read the
  mapped marker positions without a full step); a `PartialFixedProjectiveConstraint`
  (the cable's clamped root, partially free) is ignored instead of being treated
  as a full `FixedProjectiveConstraint` that removes its nodes from the state.
- `MappedStateObservationManager`: the observation mapping is optional
  (marker positions are read through the wrapper, which is what a Cosserat
  `DiscreteCosseratMapping` needs); when the observation source reports no valid
  observation the innovation is skipped and the filter predicts only
  (protocol §8.4).
- `SimulatedStateObservationSource`: `trackedObservationsValid` flag, driven
  from Python per step.
- `ROUKFilter`: after the prediction resampling the Cholesky factor of the
  reduced covariance is absorbed into the projector, so `matUinv` is reset to
  identity there. Upstream left the factor `L` in `matUinv`; the correction
  normally overwrites it, but on a prediction-only step (no observation) nothing
  does, and the next resampling would factor `L` instead of `I` and shrink the
  reduced covariance.
- `OptimParams`: stays as upstream for `transformParams=exponential`
  (`P_q0 = log(stdev)^2`), so the Python side converts the physical
  coefficient of variation: `stdev = exp(sqrt(log(1+c^2)))`, 1.341194157207 for
  c = 0.30 (protocol §6; `optimus_scene.optimus_stdev`).

## 3. Cosserat patch

`BeamHookeLawForceField` caches the section stiffness (`m_K_section`) at
`init`/`reinit`. When the estimator drives `EI`/`GI` through a Data link the
cache never changes, so every sigma point runs the same stiffness and the
parameter gain is identically zero (the symptom that stalled the earlier
attempt). The patch
[0001-BeamHookeLawForceField-track-EI-GI-internal-data.patch](../third_party/cosserat-patches/0001-BeamHookeLawForceField-track-EI-GI-internal-data.patch)
adds `trackInternalData(d_EI/d_GI/...)` and a `doUpdateInternal()` that rebuilds
the cache, i.e. the SOFA-idiomatic fix (protocol §7). Gate E checks the force
scales exactly 2.0× with a 2× parameter and returns bit-exactly when restored.

## 4. Integration into the existing pipeline

No new node, topic or model. Files:

- [optimus_scene.py](../src/cable_identification/cable_identification/optimus_scene.py):
  `build_optimus_cable(root, cfg, parameters, init_values, relative_std,
  observation_std)` inserts the Optimus components around the plant built by
  `cosserat_model.build_cable` and returns an `OptimusCable` handle
  (`set_boundary_pose`, `set_observation(points, valid)`, `estimates()`,
  `log_variances()`, `innovation_rms()`). `ROUKFilter` Simplex sigma points,
  `estimatePosition=True` (REDORD needs positions in the state),
  `estimateVelocity=False`, one boundary controller applying the *same* pose to
  every sigma point (protocol §8.2, gate H counts p+1 applications per step).
- [cable_estimator_node.py](../src/sofa_ros2_adapter/sofa_ros2_adapter/cable_estimator_node.py):
  same contract as before (`/cable/observed_markers` in; `/cable/model/*`,
  `/cable/parameter_estimate`, `/cable/innovation`, `/cable/marker_rmse`,
  `/cable/estimator_status` out). Each SOFA step is a filter step. The node
  steps the estimator to the **observation stamp**, looking the boundary pose up
  on TF at each step time, then predicts up to now without correcting. Stepping
  at wall-clock cycle time instead gave a 9 % EI bias in the pipeline test
  (the boundary the estimator saw was ~one cycle newer than the markers).
  New parameters: `initial_relative_std` (coefficient of variation per
  parameter, default 0.6), `max_steps_per_cycle` (default 20; a "behind real
  time" warning is logged when the budget is hit).
- [cosserat_model.py](../src/cable_identification/cable_identification/cosserat_model.py):
  `prepare_root(..., animation_loop=...)` so the estimator can install
  `FilteringAnimationLoop`; `CableHandles.refresh_mapping()` (see §6).
- Launch [cable_closed_loop_sim.launch.py](../src/cable_bringup/launch/cable_closed_loop_sim.launch.py)
  is unchanged: `enable_optimus:=true` starts the estimator.

## 5. Gates (all runs inside the container, `./scripts/run.sh <cmd>`)

| Gate | Command | Result |
|---|---|---|
| A compile/install | `optimus` (installer) | `libOptimus.so` + headers installed under `/opt/sofa/plugins/Optimus`, RPATH `$ORIGIN;$ORIGIN/../../../lib` |
| B factory | `optimus_smoke_test` | 7/7 components instantiated from `RequiredPlugin Optimus` |
| C 21.12 parity oracle | — | **DEFERRED** (§7) |
| D log-space prior | `cable_optimus_test` | `stdev(c=0.30)=1.341194157207`, `q0=log 0.006=-5.115995809754`, reduced variance `log(1+c²)=0.086177696`, zero/negative/non-finite init rejected |
| E parameter link | `cable_optimus_test` | force ratio 2.000000000 for 2×EI and 2×GJ, prediction shift 2.8e-4 / 2.3e-4, restore diff 0.0; EI 0.006 vs 0.010 separates markers by 51.6 mm rmse (> 3σ_obs = 6 mm) |
| H identical boundary | `cable_optimus_test` | 300 steps × (p+1 = 2) boundary applications |
| F EI recovery (0.006 → 0.010, σ_obs 2 mm) | `cable_optimus_test` | EI 0.010017 (0.17 %), moving within 50 steps, converged at step 31, log-variance 8.6e-2 → 1.8e-5, innovation 3.46 → 3.43 mm (floor 3.46 mm), held-out validation trajectory 0.22 mm (start 55.6 mm, +5 % EI 6.25 mm) |
| F/3 sparse observations (1 in 3 valid, prediction-only otherwise) | `cable_optimus_test` | EI 0.009986 (0.14 %), converged at step 34, held-out 0.17 mm |
| G GJ recovery (0.012 → 0.008, torsion ±20° @ 0.4 Hz) | `cable_optimus_test` | GJ 0.007881 (1.49 %), converged at step 132, held-out 0.36 mm (+5 % GJ 1.19 mm) |
| I pipeline, headless | `cable_optimus_pipeline --duration 60` | truth cable + synthetic markers + `cable_estimator_node`: EI 0.006 → 0.010029 (0.29 %) in 60 s, 479/480 steps corrected, median innovation 3.49 mm, `OPTIMUS_PIPELINE_TEST_PASSED` |

`cable_optimus_test` finishes in ~15 s, `cable_optimus_pipeline` in ~75 s. Both
print a single `..._PASSED`/`..._FAILED` line and exit non-zero on failure;
the pipeline test refuses to start (exit 2) when a cable stack is already
running, because two estimators on the same topics corrupt each other's traces.

## 6. Findings worth knowing before touching this

- **Plant boundary lag.** The plain plant (`cable_sofa_node`) applies a new base
  pose one solve late: `set_base_pose` writes the rigid base, but the mapped
  frames/markers are only refreshed by the next mechanical propagation. The
  estimator (Optimus pre-solve propagation) is "fresh". For a truth/estimator
  comparison inside one process call `CableHandles.refresh_mapping()`
  (`mapping.init()` re-applies the mapping; `Sofa.Simulation.updateVisual` does
  not). The plant's default was left as is because the certified artifacts were
  generated with it.
- **ROUKF state error.** Because positions are in the state, a wrong parameter
  pollutes the state estimate for a few steps after a boundary transient; large
  torsion excitations (±45°) made GJ overshoot. ±20° @ 0.4 Hz is the design
  point that recovers GJ within 1.5 %.
- **`copyStateFilter2Sofa(_, true)`** reconstructs velocity as
  (x − x_begin)/dt; exact for `EulerImplicitSolver` (verified to 1e-14).
- `FilteringAnimationLoop` propagates the root-level `AnimateBeginEvent` once;
  each sigma point's `computeSofaStep` propagates it again, hence the boundary
  controller sees p+1 events per step and must apply the *same* pose each time
  (gate H).
- `Sofa.Simulation.reset` restores `reset_position` and re-propagates mappings;
  gate E's "restore" comparison needs it (a stale mapping cache otherwise shows
  ~8e-8 differences).
- `ros2 run` children survive SIGTERM to the wrapper; the pipeline harness uses
  process groups and `kill -9` fallback. Check with
  `pgrep -fa "[c]able_estimator_node|[c]able_sofa_node|[s]ynthetic_marker_node"`.

## 7. Deferred: gate C (21.12 parity oracle)

The protocol asks for the same scene run on stock SOFA 21.12 + Optimus binary
as an oracle. Not done: a second SOFA install (~3 GB) plus its build does not
fit the 5.7 GB free disk of this host, and the 21.12 binary does not ship
Cosserat, so the oracle scene would have to be a non-Cosserat beam anyway.
Recipe when the resources exist:

1. Second container from `sofa-framework` 21.12 binary + Optimus 21.12 release.
2. Scene: 4-node `BeamFEMForceField` cantilever with `OptimParams` on
   `youngModulus` (`transformParams=exponential`, `stdev=1.341194157207`),
   `ROUKFilter` Simplex, `MappedStateObservationManager` on the tip, observations
   from a run at the true modulus + 2 mm noise (seeded).
3. Same scene on 25.12 with `third_party/Optimus`; compare the parameter trace
   and reduced covariance step by step (tolerance 1e-6 relative; the only
   allowed difference is the `matUinv` reset, which only triggers on
   prediction-only steps, so use an all-valid observation stream).

Until then the correctness evidence is gates D–I above (exact log-space
numbers, exact force scaling, recovery to < 0.3 % on EI and 1.5 % on GJ with
held-out validation, and the end-to-end pipeline).

## 8. Rebuilding after edits

- Optimus sources: `docker compose exec -T ur5e_ts_ibvs bash /ros2_ws/scripts/install_optimus.sh --force optimus`
- Cosserat patch: same with `--force cosserat` (several minutes at -j2).
- Python (`cable_identification`, `sofa_ros2_adapter`): ament_python installs
  by copy — `MAKEFLAGS=-j2 colcon build --executor sequential
  --parallel-workers 1 --packages-select cable_identification sofa_ros2_adapter`
  in an `in_live` shell, then re-run the gates.

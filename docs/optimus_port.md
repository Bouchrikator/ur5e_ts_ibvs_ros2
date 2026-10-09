# Optimus on SOFA v25.12 — port record

Status: **installed and gated**. Optimus' reduced-order UKF now runs inside the
existing `cable_estimator_node`; the in-process `LogParameterUKF` is no longer
used by the node (the module and its tests stay as a library). Written against
the portage protocol `optimus_sofa25_portage.md` (Optimus 1.1 → SOFA 25.12,
minimal core, log-space parameters, gates A–I).

**2026-10-09: the plant changed underneath the estimator** (planar extensible
Cosserat model, see [cable_dynamics_remarks.md](cable_dynamics_remarks.md) and
[status §5.17](cable_ts_status_and_diagnosis.md)): `Vec6d` strain state, Cosserat
patch 0002 (consistent Jacobian), rod-segment rotational inertia, Kelvin-Voigt
damping, no-slip grasp. The port was extended to the `Vec6` state (§2) and every
gate was re-run (§5): B, D, E, F (three variants) and H pass with the numbers
below; G (GJ) and the EI criterion of I are now **measured identifiability
limits** (§7), reported as `[LIMIT]` by the gates instead of a recovery figure.
The 2026-09 numbers of G and I depended on a mass-property bug and on an assumed
grasp compliance respectively; they are in git history only.

## 1. Pinned versions

| Component | Version | Where |
|---|---|---|
| SOFA | v25.12.00 binary (`/opt/sofa`, SofaPython3) | [docker/Dockerfile](../docker/Dockerfile) |
| Optimus | upstream `sofa-framework/Optimus` @ `cd41bf17d1ae451aa7208ed0e2219d0702847bbd`, minimal core, vendored and modified | [third_party/Optimus](../third_party/Optimus) |
| Cosserat | `SofaDefrost/Cosserat` @ `f64e029de61d320700bcb5f0ececc4efc0d8b1d9` (= the commit of the shipped `release-v25.12` binary), rebuilt from source with two patches (§3) | [third_party/cosserat-patches](../third_party/cosserat-patches), [scripts/install_optimus.sh](../scripts/install_optimus.sh) |
| Toolchain | Ubuntu 24.04, gcc 13.3, cmake 3.28, make -j2 (7 GB host) | container |

One code path builds both plugins: `scripts/install_optimus.sh` runs in the
Dockerfile stage (`WITH_OPTIMUS=true`, default) and from `./scripts/run.sh`
(`in_live` calls it before every cable command; `./scripts/run.sh optimus`
runs it alone). It is idempotent: markers are `Cosserat/.cable-cosserat-build`
(commit + sha256 of the patch set, so a changed or added patch rebuilds Cosserat once;
until 2026-10-09 the marker was the `doUpdateInternal` declaration in the installed
header) and `Optimus/lib/libOptimus.so`. The Cosserat
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
- `ObservationManager` (base): `observationVariances`, a per-coordinate variance
  vector rebuilt into `R`/`R^-1` at every observation; a non-positive or
  non-finite entry means "not observed this step" and gets `R^-1 = 0`, which is
  exactly the dropped row in the information form (`(HL)' R^-1` zeroes its
  column). Empty vector = the scalar `observationStdev` as upstream.
- `SimulatedStateObservationSource`: `trackedObservationsValid` flag, driven
  from Python per step.
- `ROUKFilter`: after the prediction resampling the Cholesky factor of the
  reduced covariance is absorbed into the projector, so `matUinv` is reset to
  identity there. Upstream left the factor `L` in `matUinv`; the correction
  normally overwrites it, but on a prediction-only step (no observation) nothing
  does, and the next resampling would factor `L` instead of `I` and shrink the
  reduced covariance.
- `ROUKFilter`: innovation gate **after the prediction**, inside the correction.
  `NIS = nu' S^-1 nu` with `S = HL U^-1 (HL)' + R`, computed by Woodbury from
  quantities the correction already has (`nu' R^-1 nu - w' U+^-1 w`,
  `w = (HL)' R^-1 nu`, unobserved rows drop out), compared with the chi-square
  quantile for the *observed* coordinates (Wilson-Hilferty, `innovationGateSigma`
  sigmas; 0 = off). A rejected observation leaves `U` and the state as predicted.
  Outputs `nis` and `correctionApplied`.
- `StochasticStateWrapper<Vec6Types, double>` and
  `MappedStateObservationManager<double, Vec6Types, Vec3Types>` (2026-10-09): the
  plant's strain state is `Vec6d` (`stateDim`: posDim = velDim = 6). The manager's
  observation data and observation source now carry the *observed* type
  `DataTypes2` (Vec3 markers) instead of the master type; the direct-mapping path
  (observations given in master coordinates) is compiled only when master and
  observed types coincide, as in the original `<double, Vec3, Vec3>`, and the
  observation vector is filled with `DataTypes2::spatial_dimensions` per point.
  `optimus_scene.py` uses `template="Vec6d"` / `"double,Vec6d,Vec3d"`, and its
  boundary controller re-syncs the Kelvin-Voigt coefficients to the sigma point's
  EI (`CableHandles.sync_damping`) because the damping is proportional to the
  stiffness the point propagates with.
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

[0002-consistent-tangent-and-vec6-strain-route.patch](../third_party/cosserat-patches/0002-consistent-tangent-and-vec6-strain-route.patch)
(2026-10-09) is a mechanics fix, not an Optimus need: the tangent operator of
`DiscreteCosseratMapping` is built from the full twist (rest axial strain 1), so
`applyJ` is the derivative of `apply()`; and the `Vec6` route (strain-rate index in
`applyJ`, `applyJT` writing to copies and zeroing the linear-strain rows, Hooke law
reading never-initialised `d_EIy/d_EIz`) is made usable. Details and evidence:
[cable_dynamics_remarks.md §4](cable_dynamics_remarks.md), `./scripts/run.sh cable_fom_test`.

## 4. Integration into the existing pipeline

No new node, topic or model. Files:

- [optimus_scene.py](../src/cable_identification/cable_identification/optimus_scene.py):
  `build_optimus_cable(root, cfg, parameters, init_values, relative_std,
  observation_std, innovation_gate_sigma)` inserts the Optimus components around
  the plant built by `cosserat_model.build_cable` and returns an `OptimusCable`
  handle (`set_boundary_pose`, `set_observation(points, variances, valid)` with
  NaN rows for unobserved markers, `estimates()`, `log_variances()`,
  `innovation_rms()`, `nis()`, `correction_applied()`). `ROUKFilter` Simplex
  sigma points, `estimatePosition=True` (REDORD needs positions in the state),
  `estimateVelocity=False`, one boundary controller applying the *same* pose to
  every sigma point (protocol §8.2, gate H counts p+1 applications per step).
  Refuses `GJ` when the configuration is planar (§7).
- [cable_estimator_node.py](../src/sofa_ros2_adapter/sofa_ros2_adapter/cable_estimator_node.py):
  same contract as before (`/cable/observed_markers` in; `/cable/model/*`,
  `/cable/parameter_estimate`, `/cable/innovation`, `/cable/marker_rmse`,
  `/cable/estimator_status` out). Each SOFA step is a filter step and **model
  time advances only by steps**: observations are queued and processed in stamp
  order, each at the step ending nearest its stamp (|error| <= dt/2), with the
  boundary pose read from TF *at every step time* (no fallback to "latest": if
  TF is not there yet the node waits) and the observation frame transform taken
  at the observation stamp. An observation the model has already passed is
  dropped as stale, never assigned to a later step; a backlog is worked off at
  `max_steps_per_cycle` per cycle with a "behind" warning, never skipped. The
  only clock jump left is loud: when the boundary history is older than the TF
  cache (10 s stall) the model is re-anchored at the observation with an error
  log. Per marker: `valid` and the covariance diagonal (rotated into the model
  frame) go to `observationVariances`; a message with some markers missing is a
  correction on the observed ones, `dof = 3 x observed`. The old Python RMS gate
  before the step is gone; the gate is the filter's NIS test (parameter
  `innovation_gate_sigma`, default 3). Stepping at wall-clock cycle time with a
  latest-TF boundary gave a 9 % EI bias in the pipeline test before all this.
  Parameters: `initial_relative_std` (coefficient of variation per parameter,
  default 0.6), `innovation_gate_sigma` (3.0), `max_steps_per_cycle` (20).
  Startup warning in the table configuration without gravity: EI is identified
  relative to the attachment springs; with the no-slip grasp the printed ratios are
  `k_g L³/EI = 3.4e6` and `k_θ L/EI = 700`, i.e. the markers carry almost no EI
  information there (§7).
- [cosserat_model.py](../src/cable_identification/cable_identification/cosserat_model.py):
  `prepare_root(..., animation_loop=...)` so the estimator can install
  `FilteringAnimationLoop`; `CableHandles.refresh_mapping()` (see §6).
- Launch [cable_closed_loop_sim.launch.py](../src/cable_bringup/launch/cable_closed_loop_sim.launch.py)
  is unchanged: `enable_optimus:=true` starts the estimator.

## 5. Gates (all runs inside the container, `./scripts/run.sh <cmd>`)

Results of 2026-10-09 (plant of [status §5.17](cable_ts_status_and_diagnosis.md),
truth EI 0.010, start 0.006, σ_obs 2 mm, 3σ NIS gate on):

| Gate | Command | Result |
|---|---|---|
| A compile/install | `optimus` (installer, `--force optimus` after the Vec6 port) | `libOptimus.so` + headers installed under `/opt/sofa/plugins/Optimus`, RPATH `$ORIGIN;$ORIGIN/../../../lib` |
| B factory | `optimus_smoke_test` | all components instantiated from `RequiredPlugin Optimus`, `OPTIMUS_SMOKE_TEST_PASSED` |
| C 21.12 parity oracle | — | **DEFERRED** (§8) |
| D log-space prior | `cable_optimus_test` | `stdev(c=0.30)=1.341194157207`, `q0=log 0.006=-5.115995809754`, reduced variance `log(1+c²)=0.086177696`, zero/negative/non-finite init rejected, GJ refused in the planar configuration, `exp(q)` = physical value after the run |
| E parameter link | `cable_optimus_test` | force ratio 2.000000000 for 2×EI and 2×GJ, prediction shift 1.6e-2 (EI) / 1.0e-1 (GJ, poses), restore diff ≤ 3e-17; EI 0.006 vs 0.010 separates the markers by 89.0 mm rms (> 3σ_obs = 6 mm); GJ 0.012 vs 0.008 by 2.27 mm = 1.1σ_obs → `[LIMIT]`, decides G |
| H identical boundary | `cable_optimus_test` | 300 steps × (p+1 = 2) boundary applications |
| F EI recovery (hanging rod, gravity, observation every step) | `cable_optimus_test` | EI 0.009986 (0.14 %), converged at step 10, log-variance 8.6e-2 → 1.1e-6, innovation 3.38 → 3.56 mm (floor 3.46 mm), NIS/dof 1.07 (last quarter), 0/300 rejected, held-out trajectory 0.32 mm (start 94.0 mm, +5 % EI 11.3 mm) |
| F/3 sparse instants (1 step in 3 observed) | `cable_optimus_test` | EI 0.009989 (0.11 %), converged at step 22, NIS/dof 0.96, 1/100 rejected, held-out 0.26 mm |
| F/occl per-marker occlusion (p = 0.3) | `cable_optimus_test` | EI 0.009996 (0.04 %), converged at step 11, NIS/dof 1.01, 2/300 rejected, held-out 0.10 mm |
| G GJ under a ±20° base roll | `cable_optimus_test` | **`[LIMIT]`, not attempted**: 1.1σ_obs separation (§7). Before the gate became conditional, the same run drifted to GJ 0.0104 (30 % off) without converging. Overall `OPTIMUS_RECOVERY_TEST_PASSED` |
| I pipeline, headless, 20 % per-marker dropout | `cable_optimus_pipeline --duration 60` | pre-check: EI 0.006 vs 0.010 separates the markers by 0.35 mm = 0.2σ_obs along this trajectory → EI criterion **`[LIMIT]`** (§7); consistency criteria met: 1789/1799 steps corrected, 10 rejected by the gate, median innovation 3.41 mm (≤ 1.5 × 3.46 mm), `OPTIMUS_PIPELINE_TEST_PASSED`. The EI estimate wanders where the data do not constrain it (0.0112 and 0.0123 in two runs) |

`cable_optimus_test` finishes in ~20 s, `cable_optimus_pipeline` in ~75 s. Both
print a single `..._PASSED`/`..._FAILED` line and exit non-zero on failure;
the pipeline test refuses to start (exit 2) when a cable stack is already
running, because two estimators on the same topics corrupt each other's traces.
The NIS/dof ≈ 1 rows are the chi-square consistency check of the filter
covariance: `S` is what the innovations actually scatter with, so the 3σ gate
rejects at the nominal 0.1–0.3 % rate.

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
  pollutes the state estimate for a few steps after a boundary transient. (The
  2026-09 "±20° @ 0.4 Hz recovers GJ within 1.5 %" relied on the frame-inertia bug,
  §7; with physical inertia GJ is not observable from that excitation.)
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
- **CPU starvation makes the truth inconsistent, and the gate says so.** The
  truth plant (`cable_sofa_node`) keeps real time by dropping physics steps
  (`solver_stats` second field); its cable then lags its own boundary and the
  estimator, which integrates every step, sees innovations the modelled `S`
  cannot explain. One pipeline run with two `ros2 topic echo` processes on the
  7 GB host rejected 347/597 observations and stalled EI at 0.0072; the same
  run unloaded rejects 4–19 of ~1200. A high `rejected` count in
  `/cable/estimator_status` together with dropped truth steps is a load problem,
  not a filter problem.

## 7. Identifiability limits (read before designing an experiment)

- **GJ is structurally unidentifiable in the table configuration.**
  `cable_common.yaml` sets `planar: true`, which the plant implements as a
  `PartialFixedProjectiveConstraint` with `fixedDirections=[1,1,0,0,0,1]` on the
  `Vec6d` strain state: torsion `kappa_x`, `kappa_y` and the shear `eps_z` are held
  at 0. With `V = 1/2 ∫ [GJ kappa_x² + EI (kappa_y² + kappa_z²) + EA eps_x² +
  GA (eps_y² + eps_z²)] ds`, `dV/dGJ = 1/2 ∫ kappa_x² ds = 0`,
  so the marker Jacobian w.r.t. `log GJ` is zero and the Fisher information is
  zero: no estimator can recover it. `build_optimus_cable` raises `ValueError`
  for `GJ` with a planar configuration (gate D checks it) instead of running a
  filter whose GJ variance can only stay at the prior.
- **Gate G (2026-10-09): GJ is not identifiable from centerline markers under a
  base roll either.** The 19.7 mm rms separation recorded on 2026-09-09 was an
  artefact of the frame mass: `UniformMass(totalMass=...)` on Rigid3d frames leaves
  `inertiaMatrix` at identity, i.e. a rotational inertia of 1.7e-3 kg m² per frame
  (the physical rod segment has 1.4e-8 axially), which turned the twist into a slow
  torsional pendulum that gravity could not follow. With the physical inertia
  (`frame_rigid_mass`) the torsional mode of this cable sits near 36 Hz, the 0.4 Hz
  roll is quasi-static, gravity's restoring moment on the sag plane (~0.2 N m)
  dominates `GJ θ / L ≈ 0.004 N m`, and a 50 % change of GJ moves the markers by
  2.27 mm rms = 1.1 σ_obs. `optimus_recovery_test` now measures that separation
  first and records G as `[LIMIT]` (no recovery attempted) when it is below 3 σ_obs.
  GJ needs an orientation observation of a cross-section or the wrist torque; the
  planar refusal (gate D) is unchanged.
- **EI in the table configuration is not identifiable from marker positions with
  the no-slip grasp (2026-10-09).** Without gravity and with the end position AND
  orientation prescribed, the quasi-static elastica shape minimises `EI·E(q)` and is
  independent of EI; only the grasp reaction scales with it. The former 0.10 %
  recovery of gate I relied on the arbitrary tip compliance `k_θ = 0.05 N m/rad`
  (`λ_r = k_θ L/EI = 3.5`), i.e. EI was identified *relative to an assumed grasp
  compliance*. With the no-slip grasp now in `cable_common.yaml` (1e5 N/m,
  10 N m/rad, `λ_r = 700`) the measured marker separation between EI 0.006 and 0.010
  along gate I's own gripper trajectory is 0.35 mm rms = 0.2 σ_obs (2.3 mm = 1.1 σ
  even with the old 0.05 N m/rad under the corrected mechanics), and the ROUKF ends
  wherever the prior and the noise leave it (EI 0.0112 and 0.0123 ± 0.0008 in two
  runs) with the innovation at the noise floor: a consistent fit of an unobservable
  parameter. `optimus_pipeline_test` measures that separation
  headlessly before spawning the nodes; below 3 σ_obs it reports `[LIMIT]` and judges
  the pipeline on its consistency (corrected steps, innovation ≤ 1.5× noise floor,
  gate rejections) instead of the EI tolerance. Options, in order of preference:
  identify EI in the hanging configuration (gate F: free tip, gravity is the
  absolute force scale, 0.04–0.14 % recovered), add the wrist wrench to the
  observation model (the grasp reaction is `EI`-proportional), or measure the real
  grasp compliance and put it in `grasp_angular_stiffness`. Do not soften the grasp
  in simulation to recover identifiability: it only encodes an assumption.
- **EA and GA are not estimated.** They are declared in `cable_common.yaml`
  (5000 N, 2000 N, nominal, uncalibrated) and shared by truth and estimator;
  `_PARAM_FIELD` only exposes EI and GJ. Under the table excitation the extension is
  ~1e-3 (tension up to ~20 N): identifying EA needs the wrist force or a traction
  test (see the measurement list in [status §7](cable_ts_status_and_diagnosis.md)).

## 8. Deferred: gate C (21.12 parity oracle)

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
   allowed differences are the `matUinv` reset, which only triggers on
   prediction-only steps, so use an all-valid observation stream, and the
   NIS gate, so run with `innovationGateSigma=0`).

Until then the correctness evidence is gates D–I above (exact log-space
numbers, exact force scaling, EI recovery to < 0.15 % in the hanging configuration
with held-out validation, NIS/dof ≈ 1, the end-to-end pipeline's consistency), and
the two measured identifiability limits of §7.

## 9. Rebuilding after edits

- Optimus sources: `docker compose exec -T ur5e_ts_ibvs bash /ros2_ws/scripts/install_optimus.sh --force optimus`
- Cosserat patch: same with `--force cosserat` (27 s on this host at -j2); a new or
  edited patch is also picked up automatically by the next `in_live` command (marker
  = patch-set hash). After a Cosserat rebuild run `cable_plugin_test`,
  `cable_forward_test` and `cable_fom_test` before the Optimus gates.
- Python (`cable_identification`, `sofa_ros2_adapter`): ament_python installs
  by copy — `MAKEFLAGS=-j2 colcon build --executor sequential
  --parallel-workers 1 --packages-select cable_identification sofa_ros2_adapter`
  in an `in_live` shell, then re-run the gates.

# Cable TS-PDC shape control — status and diagnosis

Working notes for the SOFA/Gazebo cable identification and control stack.
Written 2026-08-30. Everything below was measured, not estimated; commands to
reproduce each number are given.

---

## 1. Summary

> **2026-10-09: the SOFA plant was replaced underneath this stack** (section 5.17):
> planar extensible Cosserat model with a consistent Jacobian, physical frame
> inertia, Kelvin-Voigt damping and a no-slip grasp. The FOM is numerically
> verified (`./scripts/run.sh cable_fom_test`); the strain-POD ROM is rebuilt up to
> the order selection, which was stopped before an order was promoted
> ([sofa_mor_pipeline.md §3](sofa_mor_pipeline.md)). **Everything identified,
> certified or measured below on the plant (TS models, the `*_narrow` gains and
> certificate, closed-loop and identification numbers) refers to the previous
> plant** and has not been re-run on the new one; the `*_narrow` artifacts carry no
> plant hash, so nothing prevents `cable_qs_test` / `cable_closed_loop` from using
> them against the new plant: treat any such run as unverified. Remaining work and
> the missing physical measurements: section 7.

The plant, the observation pipeline and the online parameter identification
work and are verified. The reduced model is now good enough to be worth
certifying — the free-running rollout is inside the 50 mm gate on a matched
dataset for the first time. The **basic common-quadratic PDC LMIs are
infeasible on the identified matrices**. The earlier rejection of the relaxed
solver points was made by a defective acceptance step (it discarded the slack
`Q` and applied the basic gates — section 4.9, found by an external audit and
fixed); the 2026-08-30 rerun with the corrected verifier reports **basic and
relaxed solver-infeasible** on the regenerated 400×5 `--no-input-feedthrough`
model — a sound negative this time — while the regenerated subset table
materially changes the diagnosis (4.6): every rule PAIR now certifies, and
the obstruction localises to rules 2/3 across the parameter box. Artifacts in
`./artifacts/`; section 4 is the full diagnosis.

Current model: 3 modes, state `[q₁ q₂ q₃ q̇₁ q̇₂ q̇₃ p_gx p_gy]` (8), input
`[v_x v_y]` (2), 4 rules, `Ts = 0.04 s` (25 Hz). Envelope `EI ∈ [0.005, 0.015]`,
`rayleigh_stiffness ∈ [0.018, 0.022]`.

Four real bugs were found and fixed on 2026-08-30 (sections 5.11–5.14). Their
combined effect, measured on one fixed dataset (400 steps × 5 trajectories × 4
parameter vertices) so that only the pipeline changed:

| metric | before | after |
|---|---|---|
| basis reconstruction (3 modes) | 8.87 mm | **3.50 mm** |
| held-out one-step, marker space | 12.35 mm | **3.76 mm** |
| held-out free-running rollout | 109.5 mm | **32.5 mm** |
| `corr(q, gripper)` | 0.76 | **0.00** (guaranteed on the fitting set — 5.13) |
| 4-vertex LMI (basic) | infeasible | infeasible (relaxed too — sound gate, 4.9) |

On the shipped defaults (600 steps × 6 trajectories, a longer and wider
excitation) the same pipeline gives 3.50 mm reconstruction, 4.13 mm one-step
and 56.8 mm rollout. The rollout is the harsh metric and it is still the one to
watch.

| Plan step | Status |
|---|---|
| 1 — Validate pushed code | **Done.** Both SOFA tests pass |
| 2.1 — Grasp frame | **Not verified.** Needs a visual check in Gazebo |
| 2.2 — Explicit attach state | **Done.** `/cable/attach`, `/cable/detach`, `/cable/grasp_state` |
| 2.3 — Attachment stiffness ≤ 1 mm | **Done.** Measured 0.20 mm |
| 2.4 — Table model | **Done.** Planar constraint (was a silent no-op, see 5.1) |
| 3 — Truth / estimator split | **Done.** Verified non-circular |
| 4.1 — Synthetic observations | **Done** |
| 4.2 — Camera observations | **Done, never run against a live camera.** Now marker-free (5.15) |
| 5 — Optimus | **Done.** Ported to SOFA 25.12, drives `cable_estimator_node` (3.2, [optimus_port.md](optimus_port.md)) |
| 6 — Reduced model | **Done.** Boundary-conditioned POD split, 3 modes, 3.50 mm (naming: 5.13) |
| 7 — TS vertices from SOFA | **Done.** 16/16 cells, held-out one-step 3.76 mm |
| 8 — PDC synthesis | **BLOCKED.** Basic and relaxed solver-infeasible on the current model (sound gate, 2026-08-30 rerun) — sections 4, 4.9 |
| 9 — Outer cable TS node | **Done, but Python not C++** (plan asked for `.cpp`) |
| 10 — TS-IBVS integration | **Code done, never run** |
| 11 — Closed-loop launch | **Done.** Parses; never launched end-to-end |
| 12 — Validation order | **Runs 1–2 done, 3–14 not executed** |

Unit tests: **193 passing** (28 identification + 48 perception + 117 control).

**The control loop still cannot run**: no gains exist (section 3.1). The
runtime state mismatch of the previous revision is fixed (section 5.14).

---

## 2. What is verified working

### 2.1 SOFA plant (plan step 1)

```bash
./scripts/run.sh cable_plugin_test     # CABLE_PLUGIN_TEST_PASSED
./scripts/run.sh cable_forward_test    # CABLE_FORWARD_TEST_PASSED
```

The plugin test confirms `BeamHookeLawForceField` exposes the data fields the
estimator writes to: `EA, EI, GA, GI, rayleighStiffness`. Geometry: 31
centreline frames, markers at frame indices `[3, 8, 12, 16, 21, 26, 30]`.

Forward test, all six checks:

| check | result |
|---|---|
| zero gravity → straight | tip = `[0.5, -0.0, -0.0]` |
| gravity → sag > 5 mm | sag = 0.4390 m |
| EI ×10 reduces sag | 0.1269 m vs 0.4390 m |
| `set_parameter` + `reinit` takes effect | 0.4390 → 0.2199 |
| sag stable under refinement | 0.3 % change at 2× resolution |
| **grasped tip reaches target, base clamped** | **err = 0.20 mm** (plan wants ≤ 1 mm) |

### 2.2 Truth / estimator separation (plan step 3)

```
truth     EI = 0.0100   GJ = 0.0080     (cable_truth.yaml)
estimator EI = 0.0060   GJ = 0.0120     (cable_estimator_initial.yaml)
geometry/BC shared identically: True    (both inherit cable_common.yaml)
```

Geometry cannot drift between the two models, and the estimator starts
deliberately wrong, so the identification is not circular.

### 2.3 Parameter identification (plan step 5)

`cable_estimator_node` runs Optimus' ROUKF inside its SOFA scene
(`optimus_scene.build_optimus_cable`), parameters in log space
(`transformParams=exponential`, the plan's `q_EI = log(EI)` / `EI = exp(q_EI)`
adapter). Against the Cosserat truth plant with 2 mm marker noise, EI recovers
**0.006 → 0.010 within 0.3 %** (also with every marker occluded 30 % of the
time) and GJ **0.012 → 0.008 within 2 %** on the hanging, non-planar rod, with a
held-out trajectory check (`./scripts/run.sh cable_optimus_test`). GJ is
structurally unidentifiable in the planar table setup and refused there
(optimus_port.md §7). The earlier
in-process `LogParameterUKF` (same log-space adapter, analytic cantilever
recovery within 5 %) stays as a library with its 21 tests.

Publishes the four topics the plan asks for: `/cable/parameter_estimate`,
`/cable/innovation`, `/cable/marker_rmse`, `/cable/estimator_status`.

### 2.4 TS identification (plan steps 6–7)

```bash
./scripts/run.sh cable_dataset    # 14400 samples: 4 vertices x 6 trajectories
./scripts/run.sh cable_basis      # 3 modes, 99.29 % energy, 3.50 mm rmse
./scripts/run.sh cable_identify   # 16/16 cells
```

The dataset is sampled at the exact controller period (40 ms = 4 SOFA
substeps, zero-order hold) and split into train/validation by whole
trajectory, so the rollout below is a real contiguous prediction rather than a
walk over scattered samples. Both errors are in **marker space**: the
prediction is reconstructed through the basis and compared against the recorded
markers, so they include the representation error and are directly comparable
to the shape tolerance (see 5.12).

Model order, on a fixed 400 x 5 dataset:

| model | basis rmse | held-out one-step | blended rollout (2 s) |
|---|---|---|---|
| 2 modes | 6.73 mm | 6.50 mm | 40.96 mm |
| **3 modes** | **3.50 mm** | **3.80 mm** | **38.62 mm** |
| 4 modes | 2.72 mm | 3.38 mm | 34.72 mm |

Three modes is the default: the smallest order whose reconstruction is inside
the shape tolerance. Every `A_i` has `ρ(A) = 1.0000` exactly — the gripper
integrator sits on the unit circle and the cable modes are strictly inside.

---

## 3. What does not work

### 3.1 LMI synthesis is infeasible — blocks steps 8, 9, 10, 12

```
$ ./scripts/run.sh cable_lmi
synthesising 4 PDC gains for 4 parameter vertices (state 6, input 2)
  basic   : infeasible
  relaxed : infeasible
ERROR: the LMIs are infeasible.
```

(Console from the earlier 2-mode revision — the current model is state 8:
3 modes + velocities + gripper block. After the fixes of 5.11–5.14 the basic
problem stayed infeasible and the relaxed solver started returning points;
those points were rejected by a verification step that was itself wrong. The
2026-08-30 rerun with the corrected verifier — section 4.9 — reports basic AND
relaxed solver-infeasible on the regenerated model, so the negative stands,
now soundly.)

Full diagnosis in section 4. This blocks everything downstream:
`cable_modal_basis.yaml`, `cable_ts_model.yaml` and `cable_ts_gains.yaml` are
therefore **not shipped** in `config/` — only `cable_parameter_bounds.yaml` and
`cable_target.yaml` are. The others are products of the pipeline and
fabricating them would void the stability certificate. Consequently
`cable_closed_loop_sim.launch.py` with `enable_control:=true` cannot start.

### 3.2 Optimus is installed (ported to SOFA 25.12)

Plan step 5 is now literally satisfied. The official Optimus binary targets
SOFA 21.12, so the minimal core (ROUKF, state wrapper, observation manager,
OptimParams, filtering loop) was ported and vendored in `third_party/Optimus`,
with Cosserat rebuilt from source at the release commit plus the
`BeamHookeLawForceField` internal-data patch (without it the stiffness cache
ignores parameter changes and the filter gain is zero).
`cable_estimator_node` runs on it; the in-process `LogParameterUKF` is a
library only. Gates and numbers are in [optimus_port.md](optimus_port.md):
`./scripts/run.sh optimus_smoke_test` (factory), `cable_optimus_test` (EI
0.29 %, GJ 1.95 % recovery, per-marker occlusion, NIS/dof ≈ 1),
`cable_optimus_pipeline` (headless end-to-end with 20 % marker dropout, EI
0.11 % in 60 s). The node keeps model time (steps to each observation's stamp,
TF at the step time, stale observations dropped), weights markers by their
`valid`/covariance, and gates on the filter's NIS after prediction. Limits
(optimus_port.md §7): GJ is unidentifiable with `planar: true` and is refused;
EI in the table setup is identified relative to the modelled clamp compliance
(`k_θ L/EI = 3.5`), so on hardware calibrate the clamp or identify EI hanging
under gravity. The SOFA 21.12 parity oracle (gate C) is deferred for disk
space; the recipe is in that document.

### 3.3 Cosserat restore

Cosserat used to live only in the running container, so `docker compose down`
destroyed it. Two things changed:

* the image was rebuilt, so the pinned plugin is baked in again;
* `scripts/install_cosserat.sh` restores it into a running container from the
  same pinned URL and checksum. It is idempotent and `run.sh` calls it before
  every cable command, so a container recreated from an older image repairs
  itself in about a minute instead of needing a 7 GB rebuild.

```bash
./scripts/run.sh cosserat      # restore on demand
```

Verified by destroying the container, recreating it and watching the installer
repair it; `cable_plugin_test` and `cable_forward_test` both pass afterwards.

### 3.4 Modal reconstruction is now inside the shape tolerance

The 3-mode basis reconstructs shapes to **3.50 mm** against the 10 mm
`shape_tolerance_m` in `cable_target.yaml` (raised from 5 mm: the basis error
of 3.5 mm plus the ~4 mm detector observation noise put the sensing/model
floor near 5.3 mm RSS, and a tolerance below the floor cannot be met — the
final value must come from the measured end-to-end error distribution). The
previous 2-mode plain-PCA basis sat at 13.4 mm, which was the real accuracy
floor of the whole pipeline. The improvement is the boundary-conditioned POD
split (5.13), not a bigger basis: at the same order the split alone takes
8.87 mm to 3.50 mm.

### 3.5 Never executed

- Validation runs 3–14 (Gazebo visual cable, grasp/fixture frames, synthetic
  marker observation, EI recovery in the loop, camera observations, cascade,
  noise/dropout/e-stop). All need Gazebo up.
- `cable_estimator_node` has never run against a live SOFA + TF stack; only
  its pure filter is covered by tests.
- `cable_marker_tracker_node` has never seen a real camera image.
- The cascade (`reference_source:=cable`) has never been launched.

### 3.6 Deviations from the plan

- **Step 9 is Python**, not `src/cable_ts_control/src/cable_ts_controller.cpp`.
  Topics and behaviour match; the language does not.
- **Step 2.1 not verified**: `cable_grasp_frame` at `xyz="0.14 0 0"` was read
  from the URDF but never confirmed to sit where the fingers actually clamp. If
  it is wrong, every identified stiffness is biased.

### 3.7 Resolved: the runtime state now matches the identified model

`cable_state_reducer_node` used to publish `[q, q̇]` while the identified model
expected `[q, q̇, p_g]`, and the controller refused to run. The reducer now
reads the gripper from TF, projects the shape through the basis' boundary block
and publishes the full state. The boundary origin is stored in the basis file
(`boundary_reference`), so the offline state and the runtime state are centred
identically by construction rather than by convention. See 5.14.

---

## 4. Diagnosis: why the LMIs are infeasible

> **Retraction (2026-08-30).** Two claims in the first version of this section
> were wrong, and an external review of commit `a5ecaa3` caught both.
>
> 1. **The rollout figures near 10¹⁷ mm were invalid.** The validation mask was
>    drawn per sample, so `test` held scattered indices which were then fed to
>    the rollout as if they were consecutive. The second command did not follow
>    the first in time. Those numbers measured nothing.
> 2. **Requiring `ρ(A_i) < 1` was wrong.** PDC certifies the CLOSED loop
>    `G_ij = A_i − B_i K_j`, never the open-loop `A_i`. `A = 1.2, B = 1,
>    K = 0.5` gives a closed loop of `0.7`. This repository's own subset result
>    already disproved the gate: all 16 individual `(vertex, rule)` problems
>    were feasible, so those plants were stabilisable despite `ρ(A_i) > 1`.
>    The gate has been removed; `ρ(A)` is now reported as a diagnostic only.
>
> Everything below the retraction is the corrected analysis.

### 4.1 The original measurement (superseded)

The table below is the *old* unstructured fit, kept because the `ρ(A)` and
one-step columns are still informative. **The rollout column is invalid** — it
was computed over scattered validation indices treated as a time sequence. See
the retraction above.

| vtx | EI | damping | rule | ρ(A) | 1-step | rollout |
|---|---|---|---|---|---|---|
| 0 | 0.005 | 0.02 | 0 | 1.2562 | 0.908 mm | 1 427 mm |
| 0 | 0.005 | 0.02 | 1 | 1.0073 | 1.887 mm | 474 mm |
| 0 | 0.005 | 0.02 | 2 | 1.1186 | 0.389 mm | 1 428 mm |
| 0 | 0.005 | 0.02 | 3 | 0.9915 | 0.748 mm | 103 mm |
| 1 | 0.005 | 0.04 | 0 | 1.0207 | 0.453 mm | 200 mm |
| 1 | 0.005 | 0.04 | 1 | 1.0030 | 1.191 mm | 334 mm |
| 1 | 0.005 | 0.04 | 2 | 1.0347 | 0.361 mm | 202 mm |
| 1 | 0.005 | 0.04 | 3 | 0.9923 | 0.479 mm | 74 mm |
| 2 | 0.015 | 0.02 | 0 | 1.0204 | 0.827 mm | 178 mm |
| **2** | **0.015** | **0.02** | **1** | **3.4008** | **1.395 mm** | **3.4 × 10¹⁷ mm** |
| 2 | 0.015 | 0.02 | 2 | 1.1759 | 0.331 mm | 5 660 mm |
| 2 | 0.015 | 0.02 | 3 | 0.9915 | 0.993 mm | 123 mm |
| 3 | 0.015 | 0.04 | 0 | 0.9930 | 0.531 mm | 128 mm |
| **3** | **0.015** | **0.04** | **1** | **4.3809** | **0.393 mm** | **2.7 × 10¹⁸ mm** |
| 3 | 0.015 | 0.04 | 2 | 1.0481 | 1.098 mm | 597 mm |
| 3 | 0.015 | 0.04 | 3 | 0.9947 | 0.612 mm | 106 mm |

**11 of 16 models have ρ(A) > 1.** Regressor condition numbers across the same
cells are 12–50, so this is not raw collinearity.

### 4.2 Why that is impossible

A damped cable is passive: energy only decreases, so every discrete-time
`ρ(A)` must be < 1. At `dt = 1/30 s`, `ρ = 4.38` means the state grows more
than fourfold every control period. These are fitting artefacts, not physics.

The synthesis asks for a single `X = P⁻¹` satisfying

```
[ X    Gᵀ ]
[ G    X  ]  ⪰ 0        for all 40 blocks
```

No common quadratic Lyapunov function can exist for a family containing
ρ = 4.38. **The solver is not failing — it is correctly reporting that the
model set is uncertifiable.**

### 4.3 Structural confirmation

Subset solves isolate two independent blockers:

| subset | result |
|---|---|
| each single `(vertex, rule)` alone | **feasible** (all 16) |
| one vertex, 4 rules, diagonal blocks only | **feasible** (all 4) |
| one vertex, 4 rules, **with rule-transition terms** | **infeasible** (all 4) |
| all 4 vertices, diagonal blocks only | **infeasible** |
| any vertex pair, with cross terms | **infeasible** (all 6) |

So the rule-transition (cross) terms break a single parameter vertex, *and*
the shared `P` across the parameter box fails even without them.

### 4.4 Root cause

The acceptance gate measured only one-step error, which is nearly blind to
eigenvalue error because it predicts `x(k+1)` from the *true* `x(k)`. But the
deeper causes were in the model handed to the solver, not in the solver:

* the state was **not Markov** — the input is gripper *velocity*, so a
  displaced but stationary gripper keeps the cable deformed while contributing
  nothing to `[q, q̇]`. The rollout had no way to know where the boundary was;
* the model was fitted on **crisp cells** but executed **blended**, so the
  vertices never had to agree in the overlap regions where the cross-rule LMIs
  are evaluated;
* the offline state used a non-causal `np.gradient` while the runtime uses a
  causal low-pass — two different signals;
* the sample period was **0.03 s in the data and 0.0333 s in the model**;
* `A` and `B` were fitted freely, so nothing made the result a mechanical
  system, and neighbouring rules came out with spectral radii from 1.0 to 1.6 —
  far too different to share one Lyapunov function.

### 4.5 Corrections applied

| Fix | Effect |
|---|---|
| Trajectory-based split, full blended rollout | rollout is now a real measurement |
| Removed the `ρ(A)` gate | stabilisable vertices no longer rejected |
| Exact timing: 25 Hz = 4 SOFA substeps, zero-order hold | data and model agree |
| Shared causal velocity filter offline and online | one state signal |
| Joint fuzzy regression | fits the model that actually executes |
| Gripper position in the state | worst rollout 4303 → 622 mm |
| Structured second-order fit + exact discretisation | ρ(A) exactly 1.0; rollout → ~200 mm |
| Relaxed Tanaka-Wang conditions (`Y`, `s`) | larger feasible set |
| `worst_spectral_radius` on averaged cross terms only | stops rejecting valid certificates; diagonal-only for relaxed ones (4.9) |

A new finding along the way: **the joint fuzzy regressor is structurally rank
deficient**. Complementary triangular memberships are exactly affine in the
premises (`h₂ + h₃ = (x₁ − lo)/(hi − lo)` to 1e-16), so wherever a premise is
also a state component the `hᵢx` blocks are linearly dependent — σ_min ≈ 1e-15
with and without an affine term. The blended prediction is unique; the vertex
matrices are not. The fit therefore takes the minimum-norm solution.

### 4.6 Where it stands now (2026-08-30, after the four fixes of section 5)

Every `A_i` has `ρ(A) = 1.0000` exactly — the gripper integrator, with the
cable modes strictly inside. On the earlier 2-mode revision every
`(A_i, B_i)` pair was fully controllable (rank 6/6, PBH at every eigenvalue on
or outside the unit circle); **that audit has not yet been repeated on the
current 8-state model**, and rank alone is not enough — the redo should report
controllability-matrix singular values and a finite-horizon Gramian, not a
binary rank.

Minimal infeasible subsets on the 3-mode model
(`src/cable_ts_control/diag_lmi.py`), regenerated 2026-08-30 with the
corrected acceptance step of 4.9 on the regenerated 400×5
`--no-input-feedthrough` model. The earlier table, produced with the
defective gate, had misclassified three of the six rule pairs:

| subset | sound gate (2026-08-30) | previous (unsound) |
|---|---|---|
| one parameter vertex, 4 rules, cross terms | no, all four | no, all four |
| one rule, all four parameter vertices | rules 0 and 1 certify | rule 0 only |
| rule pairs inside vertex 0 | **all six certify** | three of six |

No rule PAIR is an obstruction any more: infeasibility needs at least three
rules of one vertex, and rules 2/3 cannot share a Lyapunov matrix across the
parameter box even alone. Measured per-vertex rule spread on this model:
`‖ΔA‖/‖A‖ = 0.9–2.9`, `‖ΔB‖/‖B‖ = 0.8–3.4`.

Four candidate causes were tested and **eliminated by measurement**:

| candidate | test | result |
|---|---|---|
| parameter box too wide | `EI ∈ [0.008, 0.012]`, ±20 % instead of 3:1 | still infeasible |
| premise box too wide | `--premise-margin` 1.0 → 0.3 | still infeasible at every width |
| velocity-filter lag | `--velocity-alpha` 0.4 → 1.0 | still infeasible |
| decay margin too tight | `decay` 0.999 → 0.95 | still infeasible |

What is left is the spread of the rule **input** matrices:
`max‖Bᵢ−Bⱼ‖ / ‖B‖ = 1.6 … 3.3`. The rules disagree about the input gain by
more than the gain itself, and no single `K_j` can serve them all. (Caveats:
this norm ratio is scale-dependent, blows up when `‖B‖` is small, does not
separate magnitude spread from a rotation of the input subspace, and carries
no confidence interval — the principal angles between the `Im(Bᵢ)` and a
bootstrap over trajectories are the missing measurements.) Section 5.14
identifies the mechanism (the `G u` column is nearly collinear with the
`D q̇` column, so each rule splits them differently) and shows that removing
that term costs nothing on held-out data — which demonstrates
*non-identifiability under this excitation*, not that the physical feedthrough
is zero. Dropping it moved the 3-mode problem from *infeasible* to *the solver
returns a point* on the earlier dataset generation — a point then rejected by
the defective verifier of section 4.9. The 2026-08-30 rerun answers the
question for the regenerated model: `basic: infeasible, relaxed: infeasible`
at solver level, so there is no returned point to verify and the relaxation
does not rescue the current vertices.

### 4.7 What is still open

**The synthesis LMIs match the book; the acceptance step did not.** The solve
step was checked line by line against Tanaka & Wang (2001): the basic
conditions match (3.19)–(3.20) and the relaxed ones match (3.27)–(3.28).
Taking the Schur complement of the implemented blocks returns
`G'PG − P + (s−1)PYP < 0` and `H'PH − P − PYP ≤ 0`, i.e. exactly Theorem 10
with `Q = PYP`. But the verification then threw `Y` away and checked the
BASIC inequalities — section 4.9 — so every "relaxed point rejected" statement
below says nothing about the relaxed theorem.

Hypotheses tested, all on the corrected model:

| Hypothesis | Result |
|---|---|
| Bad state scaling | **no** — three similarity transforms, all still infeasible |
| Not enough gain freedom | **no** — parameters as premises (16 rules, 16 gains) still infeasible |
| Uncontrollable mode | **no** — every vertex rank 6/6 on the 2-mode revision (8-state redo pending) |
| Rules too dissimilar | **yes**, and it is `B` rather than `A` |

Two premise choices were compared (`--premise-map`). `modal` uses the leading
modal coordinates, which *are* state components; `boundary` uses the gripper's
foreshortening `1 − ‖p_g‖/L` and bearing about the clamp, which is where the
geometric nonlinearity of a clamped inextensible rod lives and which no state
component duplicates. Neither makes the LMI feasible, and their held-out errors
are within 0.1 mm of each other. `boundary` is the default because it is the
construction Tanaka & Wang prescribe — the premises are the plant's bounded
nonlinearities — not because it measured better.

A coherence penalty shrinking each rule towards the mean rule (`--coherence`)
confirms the mechanism, and does produce a certified controller:

| coherence | rule spread ‖ΔA‖ | 4-vertex LMI | free-running error |
|---|---|---|---|
| 0 | 1.92 | infeasible | 289 mm |
| 1 | 0.12 | infeasible | 833 mm |
| 10 | **0.02** | **certified**, `max eig(G'PG−P) = −6.5e-06`, `ρ = 0.9952` | 938 mm |

**This is not the fix.** At coherence 10 the rules are nearly identical, so the
TS model has collapsed to a single LTI system — the fuzzy structure that
justifies PDC has been regularised away — and the free-running error more than
triples. The one-step error is flat across the whole sweep, so the penalty buys
certifiability without buying accuracy. It is off by default for that reason: a
diagnostic, not a solution.

### 4.8 A margin bug that made the solver lie

Reaching the result above exposed a real bug in the synthesis. The margin was
an absolute floor on the Schur block, `[[X, G'], [G, X]] ⪰ eps·I`, applied in
`X = P⁻¹` space. Verification happens in `P` space, where that margin returns
as `P (margin) P` — and since `X ⪰ I` forces `P ⪯ I`, it shrinks away as `X`
grows. The solver therefore reported *feasible* on gains whose certificate then
failed:

```
basic   : feasible
max eig(G' P G - P) = 4.342e-09 >= 0 FAILED
```

Replaced with the book's decay-rate conditions (Tanaka & Wang 3.36–3.37, and
3.43–3.44 relaxed), which put `βX` in the top-left block and so impose the
*relative* margin `ΔV ≤ (β − 1)V`. Being relative, it survives the change of
variables:

```
basic   : feasible
max eig(G' P G - P) = -6.549e-06 < 0 OK
worst |eig(A_i - B_i K_j)| = 0.995169 < 1 OK
```

Two regression tests now assert that a "feasible" report always implies a
strictly negative certificate, and that `decay` really bounds the decrease.

### 4.9 The relaxed verifier discarded Q — external audit finding, fixed

> Found by an external mathematical audit of commit `b1ea940` (2026-08-30).
> This is the second verification defect after 4.8, and the more consequential
> one: it did not make the solver lie, it made the acceptance step reject
> valid answers.

`solve_cable_ts_pdc` built the relaxed slack `Y ⪰ 0` into the LMIs and then
returned only `(gains, P)`. Both acceptance checks —
`verify_lyapunov_decrease` (`G'PG − P`) and `worst_spectral_radius` over the
averaged cross terms — are gates for the **basic** theorem. For a relaxed
certificate the solved inequalities are, with `Q = PYP`:

```
Gᵢᵢ' P Gᵢᵢ − βP + (s−1)Q ≺ 0
Hᵢⱼ' P Hᵢⱼ − βP − Q      ⪯ 0
```

An averaged cross term may have `ρ(Hᵢⱼ) > 1` and still be covered, because the
`(s−1)Q` surplus on the diagonal terms compensates it. Scalar counterexample
(now a permanent regression test): `P=1, Q=0.5, s=2, β=0.999, Gᵢᵢ=0,
Hᵢⱼ=1.1` — relaxed residuals `−0.499` and `−0.289`, both valid, while the old
gates compute `1.1² − 1 = 0.21 > 0` and `ρ = 1.1 > 1` and wrongly reject.

**Consequences for earlier claims in this document:**

* every "the relaxed solver returns a point that fails verification"
  statement (4.6, and the `--no-input-feedthrough` result in 5.14) is
  unsound — those points were never checked against the relaxed theorem;
* the minimal-infeasible-subset table in 4.6 was produced by `diag_lmi.py`
  using the same wrong gate and must be regenerated;
* the *basic*-condition infeasibility results are unaffected — they come from
  the solver's own status on the basic problem, which was checked with the
  matching basic gates.

**Fix (all in `lmi_synthesis.py`, with regression tests):**

* the solver returns a complete `PdcCertificate`: `Kᵢ`, `P`, `Q = PYP`, `β`,
  `s`, certificate type, solver name and status — nothing implicit;
* `verify_certificate` checks the inequalities the certificate actually
  claims. The acceptance gate is the `β' = 1` (pure stability) form, which the
  solved decay margin `β < 1` backs with real headroom against solver
  round-off; the exact at-`β` residuals are also computed and archived;
* the individual spectral-radius gate is restricted to the diagonal terms for
  relaxed certificates (their condition does imply `ρ(Gᵢᵢ) < 1`; the cross
  terms are covered by `Q`, not individually);
* `solve_cable_ts_lmi` writes the full certificate — including `Q`, `β`, `s`,
  solver status and all residuals — into the gains YAML, and the pipeline
  artifacts now persist under `./artifacts/` on the host so a run can be
  archived and independently re-verified.

**Rerun result (2026-08-30):** dataset → basis → identify
(`--no-input-feedthrough`, 400×5, reproducing the documented 3.80 mm one-step
and 32.0 mm rollout) → LMI, all artifacts archived in `./artifacts/`. The
corrected pipeline reports `basic: infeasible, relaxed: infeasible` from the
solver itself — a sound negative. The subset table (4.6) regenerated with the
sound gate DID change materially — all six rule pairs certify, rules 0 and 1
certify across the whole parameter box — so the defect had been distorting
the diagnosis, but repairing it does not, on its own, make the current model
certifiable. The obstruction is now precisely located: rules 2/3 across the
parameter box, and ≥3-rule interactions within a vertex.

The remaining work is listed in section 7.

---

## 5. Bugs found and fixed

Recorded because each was silent and each invalidated results upstream.

### 5.1 Planar constraint was a silent no-op

`PartialFixedProjectiveConstraint` was attached to `FramesMO`, which is a
*mapped* state. SOFA rejected it every step:

```
[ERROR] [MappingGraph] Requested mechanical state (.../FramesMO) is probably
mapped or unknown from the graph: only main mechanical states have an
associated submatrix in the global matrix
```

Plan step 2.4 was therefore never in effect. Fixed by moving the constraint to
the independent Cosserat strain DOFs with `fixedDirections = [1, 1, 0]`
(torsion and out-of-plane bending fixed, in-plane bending free).

### 5.2 The excitation was buckling the rod

The gripper was driven radially inwards, compressing an inextensible 0.7 m rod
until it buckled and snapped through — tip travelled `x: 0.700 → -0.088`. A
bifurcation cannot be fitted by a smooth local model. Fixed with a radial
annulus guard (`--radial-min`, default 0.92 of rod length). Modal
reconstruction error improved 15.25 → 6.21 mm on the dataset of the day; the
current 13.4 mm is a different, wider excitation (section 5.7), not a
regression.

### 5.3 Each parameter vertex got a different excitation

The RNG advanced between vertices, so parameter effects were confounded with
trajectory differences. All vertices now share one trajectory.

### 5.4 `one_step_rmse` mixed metres with metres per second

It averaged over the whole state `[q, q̇]` and the result was printed as "mm".
The number was dimensionless nonsense dominated by the velocity term. Adding
`position_dim` took the reported error from **17–47 mm to 0.49–0.97 mm** —
the physics had been fine all along.

### 5.5 The cable body was the same colour as a marker

The Gazebo cable was orange `(0.85, 0.35, 0.05)` → OpenCV hue ≈ 11, which sits
inside the tracker's orange marker window `[11, 22]`. The cable body would have
been detected as marker 1. Body is now desaturated grey; markers use seven
distinct hues matching `marker_hsv_ranges`.

### 5.6 `cable_identification` tests never ran

`colcon test` reported `NO TESTS RAN` (exit 5) because the package lacked
`tests_require=["pytest"]` and fell back to unittest discovery. Test count went
**84 → 112** once fixed.

### 5.7 Cartesian random walk left rule cells empty

Uneven angular coverage of the reachable arc left cells with as few as 3
samples. Replaced with a polar sweep (`excitation_path`), which gives 283–444
samples in all 16 cells.

### 5.8 Log-parameter underflow in the UKF

`exp(q)` underflows to `0.0` near `q = −745`, silently turning a stiffness into
zero and poisoning every later prediction. Fixed with a `|q| ≤ 50` rail plus
the innovation gate the plan requires.

### 5.9 The LMI margin was scale-dependent

The solver reported *feasible* on gains whose certificate then failed, because
the margin was an absolute floor imposed in `X` space and verified in `P`
space. Full account in section 4.8. Two regression tests now assert that
"feasible" always implies a strictly negative certificate.

### 5.10 `n_modes` was inferred as `state_dim // 2`

True only for the plain `[q, q̇]` state. Once the gripper position joined the
state it returned 3 for a 2-mode model, mis-sizing every shape target. The
mode count is now stored explicitly in `cable_ts_model.yaml` and the controller
validates the reduced state it receives (section 3.7).

### 5.11 The recorded command was one sample out of step

`generate_sofa_dataset` stored, at index `k`, the velocity that had led *into*
sample `k`, while `x(k+1) = A x(k) + B u(k)` needs the one that drives `k` to
`k+1`. The gripper is a pure integrator of the command, so the dataset carries
an exact check of its own alignment:

```
max |(g[k+1]-g[k])/dt - u[k]|   = 6.705 m/s     <- the model's assumption
max |(g[k+1]-g[k])/dt - u[k+1]| = 0.000 m/s     <- one-sample shift
command rms                     = 0.275 m/s
```

The error is 24× the command itself. It was invisible in the one-step metric
because `u` is strongly autocorrelated, but it destroys `B` and makes every
free rollout integrate a gripper trajectory that never happened. Fixing it took
the 2-mode rollout from 109.5 mm to 67.8 mm before any other change.

`identify_ts_vertices` now refuses a misaligned dataset and names the shift.

### 5.12 The acceptance metric was not comparable across model orders

The held-out error was an RMS over the *modal coordinates*. Those are divided
by the number of modes, so the reported figure shrinks when modes are added
even if the physical error grows — and the model order was being chosen on it.
At 2 versus 4 modes the modal figure said 4.5 mm versus 11.4 mm (add modes, get
worse) while the marker-space figure said 12.4 mm versus 6.9 mm (add modes, get
better). The two disagree about the direction.

Both errors are now measured in marker space: the prediction is reconstructed
through the basis and compared against the recorded markers, so the number
includes the representation error and can be compared to the shape tolerance.

### 5.13 The reduced state contained a copy of the gripper

The marker set includes `s/L = 1.0`, the tip, which the gripper holds. A plain
PCA of the shape therefore produces modal coordinates that are partly an
algebraic function of the gripper position — measured `corr(q, p_g) = 0.7577`,
and `p_g` is predicted from `q` alone with `R² = 0.954`. The state was not
minimal, part of the modal acceleration was really the gripper's, and the fit
had to absorb that somewhere.

Replaced with a boundary-conditioned POD: a regressed boundary block plus PCA
modes of the residual (Craig-Bampton-like in structure only — see the naming
note below):

```
y = y0 + Psi g + Phi q
```

`Psi g` is the quasi-static response to the prescribed boundary; `q` describes
only what the boundary does not explain, and is orthogonal to `g` by
construction. At equal model order:

| | plain PCA | boundary-conditioned POD |
|---|---|---|
| reconstruction, 3 modes | 8.87 mm | **3.50 mm** |
| `corr(q, gripper)` | 0.7577 | **0.0000** |
| held-out one-step | 6.56 mm | **3.80 mm** |
| held-out rollout | 95.3 mm | **38.6 mm** |

`ModalBasis` carries the optional `psi` block and the `boundary_reference` it
is measured from; `psi = None` reproduces the old plain PCA exactly.

**Naming, precisely** (this section used to say "Craig-Bampton"): a physical
Craig-Bampton reduction takes `Psi = −Kᵢᵢ⁻¹Kᵢᵦ` from partitioned stiffness and
`Phi` from the fixed-interface eigenproblem, with mass orthogonality. Here
`Psi` is an ordinary least-squares regression on data and `Phi` a PCA of the
residual — a *boundary-conditioned POD*. Two honesty notes follow: the
`corr(q, g) = 0` above is guaranteed **on the fitting set** by least squares
(residuals are orthogonal to regressors), so it is evidence of a better basis,
not proof of dynamic or statistical independence; and the basis is currently
built from **all** trajectories before the train/validation split, which leaks
validation information into the basis and makes held-out errors mildly
optimistic (section 7).

### 5.14 The direct input feedthrough is not identifiable

The structured fit estimates a `G u` term in the modal acceleration. With the
boundary block in place the gripper acts on the cable through its *position*,
which is already the state `g`; the grasp damper acts on a relative velocity
whose cable part is already `q̇`. What is left for `G` is nearly collinear with
the `D q̇` column, so the two trade off freely and every rule picks a different
split. That is what drives `‖ΔB‖/‖B‖` to 1.6–3.3.

Dropping the term (`--no-input-feedthrough`) costs nothing on held-out data and
*improves* the rollout:

| | with `G u` | without |
|---|---|---|
| one-step, 3 modes | 3.80 mm | 3.76 mm |
| rollout, 3 modes | 38.6 mm | 32.5 mm |
| one-step, 4 modes | 3.38 mm | 3.37 mm |

It is left opt-in: dropping it is a constitutive claim about the plant, and the
default should not silently make one. The measurement is recorded here so the
claim can be argued rather than assumed.

### 5.15 The runtime state and the marker-free observation

`cable_state_reducer_node` now reads the gripper from TF, projects through the
boundary block and publishes `[q, q̇, g]`, so the runtime state matches the
identified model (this was section 3.7). The boundary origin comes from the
basis file, so the two cannot be centred differently.

Separately, the coloured spheres are gone from the Gazebo cable
(`marker_s_over_l` now defaults to empty). They were a simulation artefact: a
real cable carries no fiducials, so a pipeline that depends on them cannot
transfer. `cable_dlo_detector_node` recovers the centreline from the cable
itself following Keipour, Bandari and Schaal, *Deformable One-Dimensional
Object Detection for Routing and Manipulation*, RA-L 2022 (arXiv:2201.06775):
segmentation → Zhang-Suen thinning → skeleton branches → fixed-length chain
fitting → cost-based merging with a smooth gap fill → arc-length resampling.
Identity, which the colours used to supply, now comes from arc length measured
from the clamped end. The node publishes the same `/cable/observed_markers`
contract, so nothing downstream changes. 22 unit tests, including an occluded
cable: a 12-pixel gap is bridged and the detected chain comes to 120.0 against
an analytic centreline of 124.0.

Which of the two the pipeline uses is the "discrete features versus continuous
curve" choice of Cuiral-Zueco and López-Nicolás, *Taxonomy of Deformable Object
Shape Control*, RA-L 2024 (§II-C). Both are kept; the marker-free one is the
default because it is the one that can run on hardware.

### 5.16 The tip spring destabilized the plant whenever it was under load

The first live run of the full stack (Gazebo + SOFA + camera DLO) showed the
truth cable slowly coiling into a 5 cm clump at the fixture — chain length
0.266 m for a 0.7 m inextensible rod, i.e. per-section curvatures near 1000
rad/m: a diverged state, not an equilibrium.

Mechanism: the tip attachment is a `RestShapeSpringsForceField` on `FramesMO`,
a *mapped* state. Forces map back through `Jᵀ`, but the spring's **geometric
stiffness never reaches the implicit solver** (the same class of problem as
5.1, on the force side instead of the constraint side). Any sustained spring
load therefore pumps energy each step. Three separate manifestations, each
reproduced headlessly and fixed:

1. **Detached hold.** The target used to pin the free tip at the analytic
   straight pose forever. A/B repro, 600 s: pinned → chain 0.31 m,
   `max|strain| ≈ 1000`; released → chain 0.7000 m, strain 0.000. Fix: the
   coupling now disables the spring while `DETACHED` (a detached cable end is
   free — the pin was unphysical anyway) and re-engages it on the latch edge,
   with the target synced to the tip so it starts at rest.
2. **Stiffness scale.** `grasp_stiffness` was 2·10⁴ N/m against a rod whose
   transverse forces are millinewtons. Swept attached holds (300 s, bent
   pose): 2·10³ N/m gives 0.21 mm attachment error (spec ≤ 1 mm) and zero
   drift; 2·10⁴ coils the rod within seconds of dragging. Angular stiffness
   went 500 → 0.05 Nm/rad for the same reason (tip moments are `EI·κ` ~ mNm;
   holding the latch orientation at 5·10² stored ~1 Nm and pushed mm-level
   position error).
3. **Over-extension.** Driving the gripper past the rod's reach (the live
   failure: a pan drag put the anchor 0.713 m from the fixture) leaves the
   spring in *sustained tension*, which flips transverse modes unstable and
   re-coils the rod. Physically, fingers pulling a taut cable slip. The
   coupling now projects the anchor into 99 % of the reachable disk
   (`_clamp_reachable`). Note the closed-loop target `q* = 0` (straight) sits
   exactly on this boundary, so transient over-extension is not an edge case —
   it is the operating point.

Validation: 5-phase headless suite (detached 120 s hold, latch, 10 cm ramped
drag with settling trace, 5 cm over-extension for 30 s, detach + 60 s
relaxation) — chain length 0.7000 ± 0.0005 m throughout, attachment error
2.6 mm at 97 % extension; `forward_test` 6/6 (grasped-tip error 0.46 mm < 1 mm
at the new stiffness); 193/193 package tests; live end-to-end: pose → explicit
attach → over-extending drag leaves chain at 0.7000 m with 7/7 valid DLO
markers on the bent cable.

Consequence for the identified artifacts: the dataset was generated with the
old 2·10⁴ spring but in short (24 s) *attached, interior-workspace* rollouts
where the instability had no time to express itself and the attachment error
(0.02 mm) was negligible; with 2·10³ that error is 0.21 mm against cm-scale
excitation — a ~1 % perturbation of the input coupling, second order next to
the model's 3.8 mm one-step residual. The dataset is not invalidated; regard
regeneration as hygiene for the next identification pass, not a blocker.

**Root cause found (2026-10-09, see 5.17):** the pinned Cosserat plugin's `applyJ`
was not the derivative of `apply()`, so the mapped spring force was not the gradient
of the spring potential and did net work over cycles. With `cosserat-patches/0002`
the same 600 s holds (bending, and sustained tension at 1.002 L) are stable with a
1e5 N/m / 10 N m/rad grasp, and the 99 % reach clamp is gone.

One operational lesson recorded alongside: during the debugging a *zombie*
truth node from a half-killed earlier launch kept publishing its coiled state
onto `/cable/truth/frames`, interleaved with the healthy publisher — the chain
metric flapped 0.70 / 0.27 m at ~1 Hz. Two publishers on one topic is silent
in ROS 2; `ros2 topic info -v` (publisher count) is the discriminating probe,
and stack restarts must verify `pgrep -f cable_sofa_node` returns exactly one.

### 5.17 The mechanical model was inconsistent in five places (2026-10-09)

Found while making the cable extensible (prompt: planar extensible Cosserat model +
POD). Each is a plugin or scene defect, verified by `cable_fom_test` and the
`cable_dynamics` identities after the fix; details in
[cable_dynamics_remarks.md](cable_dynamics_remarks.md) and `cosserat-patches/0002`.

1. **Tangent operator without the axial strain** (`BaseCosseratMapping::computeTangExp`
   padded the twist with `(0,0,0)` while `buildXiHat` used `(1,0,0)`): `applyJ` rotated
   the distal rod about the section start node (lever `l (L - s_i)` instead of
   `l (L - s_i - l/2)`), so `J^T lambda`, `J^T K_g J` and `J^T M J` were not the virtual
   work of the frames. This is the mechanism behind 5.16. Fixed: FD agreement 1.5e-10,
   `J^T lambda = -grad V_grasp` to 2.7e-8.
2. **The Vec6 route of the pinned release was unusable**: `applyJ` wrote the strain
   rate into component `i` (section index) instead of `u`; `applyJT` copied its output
   accessors by value (no force reached the inputs) and projected the linear-strain rows
   to zero (3-row selector); the Hooke law read `d_EIy/d_EIz`, Data that were declared
   but never initialised (zero bending stiffness). Fixed in the same patch; the scene
   is now `Vec6d` with finite EA and GA (traction `dL = F L/EA` to 7e-5).
3. **Rotational inertia equal to the mass**: `UniformMass(totalMass=...)` on Rigid3d
   frames leaves `inertiaMatrix` at identity, 1.7e-3 kg m² per frame against the
   physical 5e-8 (rod segment). Fixed with `vertexMass=frame_rigid_mass(cfg)`; this is
   what made GJ look identifiable under a base roll (optimus_port.md §7).
4. **Damping was not a force**: the Hooke law's `rayleighStiffness` only damps the
   velocity increment on the matrix side and the solver's Rayleigh factors damp the
   clamp and grasp springs too. Now `DiagonalVelocityDampingForceField` applies the
   Kelvin-Voigt law `rayleigh_stiffness_s diag(Sigma) l q'` on the strains (implicit
   through `B`) and `rayleigh_mass_per_s = 0` (table friction neglected). The
   convective inertia of the mapped masses `-m (dJ/dt) q'` (dropped by SOFA; 0.2 %
   median, 3 % max of the elastic force along the excitation, 5 % in a fast free
   oscillation) is available as an explicit frame wrench (`convective_inertia`) but
   **off by default**: explicit at h = 0.01 s it diverged once when the softest cable
   snapped taut, and is stable at h = 0.005 s. Implicit Euler at h = 0.01 s still
   removes 0.44x the Kelvin-Voigt dissipation (measured, first order in h).
5. **Grasp**: the target was the gripper position (the latch offset was discarded)
   with an arbitrary 0.05 N m/rad compliance and a 99 % reach clamp. Now the frame
   nearest to the gripper is grasped, the relative pose met at latch is kept (no ramp),
   pulling past the rest length stretches the rod, and the no-slip springs are
   1e5 N/m / 10 N m/rad until the real mounting is measured. Consequence for the
   identification: with a rigid grasp and no gravity the quasi-static shape does not
   depend on EI (0.2 sigma_obs separation along gate I's trajectory), so EI must be
   identified hanging under gravity or from the grasp wrench (optimus_port.md §7).

The POD chain was rebuilt on this model with the strain-energy inner product and an
independent test trajectory ([sofa_mor_pipeline.md](sofa_mor_pipeline.md)).

---

## 6. Running it

Prerequisite: the container must be up and Cosserat present (section 3.3).

```bash
docker compose ps                     # expect ur5e_ts_ibvs running
docker compose exec ur5e_ts_ibvs bash -c \
  'test -f /opt/sofa/plugins/Cosserat/lib/libCosserat.so && echo OK'
```

### 6.1 Plant tests — validation items 1–2, no Gazebo

```bash
./scripts/run.sh cable_plugin_test     # CABLE_PLUGIN_TEST_PASSED
./scripts/run.sh cable_forward_test    # CABLE_FORWARD_TEST_PASSED
```

### 6.2 Gazebo + SOFA cable — validation items 3–4

```bash
./scripts/run.sh cable_sim
```

This brings up Gazebo *and* the runSofa window, both driving the same cable:
the SOFA scene is the truth plant and publishes `/cable/truth/frames`,
`/cable/truth/markers` and `/cable/grasp_state`, and the Gazebo cable follows
those frames. Add `sofa_gui:=false` to run the plant headless instead.

In a second terminal:

```bash
docker compose exec ur5e_ts_ibvs bash
source /ros2_ws/install/setup.bash

ros2 topic hz /cable/truth/frames
ros2 topic echo --once /cable/truth/markers
ros2 run tf2_ros tf2_echo base_link cable_grasp_frame
ros2 run tf2_ros tf2_echo base_link cable_fixture_frame
ros2 run rqt_image_view rqt_image_view          # /overview/image
```

Confirm: UR5e visible, cable visible as a plain uniform cable, fixture static,
cable follows the gripper after attachment. This is also where step 2.1 (the
grasp frame) finally gets checked.

Measured on 2026-08-30: `/cable/truth/frames` at 60 Hz from the runSofa scene,
40 cable segments and 0 marker spheres spawned in `ibvs_world`.

The coloured fiducials are opt-in now that detection is marker-free (5.15):
set `marker_s_over_l` on `cable_visual_node` to the arc-length fractions you
want spheres at (it defaults to empty). The closed-loop launch wires the
marker-free `cable_dlo_detector_node` for `observation_source:=camera`;
the legacy HSV tracker remains available as
`observation_source:=camera_markers`.

Drive the arm and attach:

```bash
./scripts/run.sh pose cable
docker compose exec ur5e_ts_ibvs bash -c \
  'source /ros2_ws/install/setup.bash && \
   ros2 service call /cable/attach std_srvs/srv/Trigger'
```

### 6.3 Identification milestone — validation items 5–6

Two SOFA models, noisy synthetic markers, online EI identification, no
controller. This is the milestone the plan calls the important one.

```bash
./scripts/run.sh cable_identify_sim
```

```bash
ros2 topic echo /cable/parameter_estimate    # EI should climb 0.006 -> 0.010
ros2 topic echo /cable/innovation
ros2 topic echo /cable/estimator_status
```

Metrics are written to `~/ibvs_logs/cable_metrics.csv`.

### 6.4 Offline pipeline — headless

```bash
./scripts/run.sh cable_dataset      # ~10 min of SOFA
./scripts/run.sh cable_basis
./scripts/run.sh cable_identify     # passes one-step, fails the rollout gate
./scripts/run.sh cable_lmi          # infeasible (sound gate, 4.9); writes nothing
```

All four artifacts (`cable_dataset.npz`, `cable_modal_basis.yaml`,
`cable_ts_model.yaml`, `cable_ts_gains.yaml`) now land in `./artifacts/` on
the host, so a run survives the container and can be archived with the commit
that produced it. Every scientific claim should ship those files. The
2026-08-30 rerun kept both configurations: the shipped 600×6 defaults and the
documented 400×5 set (`*_400x5*`, identified with `--no-input-feedthrough`).
Extra flags pass through, e.g.:

```bash
./scripts/run.sh cable_identify --no-input-feedthrough \
  --dataset /ros2_ws/artifacts/cable_dataset_400x5.npz \
  --basis /ros2_ws/artifacts/cable_modal_basis_400x5.yaml \
  --output /ros2_ws/artifacts/cable_ts_model_400x5_nofeed.yaml
```

### 6.5 Unit tests

```bash
./scripts/run.sh cable_unit_tests   # 193 tests
```

### 6.6 What cannot be run

```bash
./scripts/run.sh cable_closed_loop enable_control:=true   # no gains exist
```

Blocked on gains: `cable_ts_gains.yaml` has never been produced (sections 3.1
and 4.9). The state mismatch of the previous revision is fixed (3.7), so the
gains are now the single blocker. Validation items 7–14 stay unreachable until
the pipeline rerun with the corrected verifier.

### 6.7 Build notes

The default 16-way parallelism OOMs on an 8 GB host:

```bash
MAKEFLAGS=-j2 colcon build --executor sequential --parallel-workers 1
```

`ament_python` installs by copy, not symlink: rebuild the package before
running pytest, or the stale installed copy is tested.

---

## 7. Next steps, in order

> **2026-10-09, mechanical model and POD (work stopped on request; do these first).**
> State: FOM implemented and numerically verified (`cable_plugin_test`,
> `cable_forward_test`, `cable_fom_test` 12/12, `cable_dynamics`); Optimus ported to
> the Vec6 state and gated (GJ and table-EI are measured identifiability limits,
> [optimus_port.md §7](optimus_port.md)); snapshots and weighted POD regenerated
> (r = 5..48); order selection stopped during r = 24, nothing promoted. Nothing is
> experimentally validated: every parameter is nominal.
>
> 1. Finish `./scripts/run.sh cable_sofa_mor_validate` (restarts at r = 5, then holds
>    and projected rollouts for the first passing order), evaluate the test trajectory
>    6 once, report a failure as such. Decide explicitly, before or after, whether the
>    5 % per-component tolerance on the in-plane shear is the mechanical requirement
>    (it decides the order from r = 7 to 18 while positions are at the micrometre
>    level) or whether a targeted shear-free comparison justifies dropping shear;
>    do not change it to make an order pass.
> 2. On the promoted basis: `cable_sofa_mor_plugin_test --cable-modes ...`,
>    `cable_dynamics --rom` (Galerkin identities, `Phi^T W Phi = I`,
>    `B_r = Phi^T B Phi`).
> 3. Fix the dump's straight-rod probe that leaves 1e-4 in the locked strains of the
>    dumped state ([sofa_mor_pipeline.md §9](sofa_mor_pipeline.md)), and add a
>    locked-strain == 0 assertion.
> 4. Out of this change's scope, required before any closed-loop claim on the new
>    plant: regenerate the ROM dataset, modal TS, observer, LMI and PDC stages (they
>    refuse the old basis by hash/dimension), and re-identify or retire the legacy
>    `*_narrow` TS/gains (identified on the previous plant, no plant hash).
>
> **Physical measurements needed (none exist; all values are nominal):**
> - `EI` (and its spread): hanging cantilever sag or a three-point bend; the table
>   setup with a rigid grasp cannot give it from marker positions (optimus_port §7).
> - `EA`: traction test, `dL = F L0 / EA` in the linear range (`cable_fom_test`
>   reproduces the formula to 7e-5); also the strain range where it stays linear.
> - `GA` (in-plane shear) or evidence that shear is negligible for this cable.
> - Internal damping: free-oscillation log decrement of a cantilever at two
>   amplitudes (Kelvin-Voigt `c_R`), and the table friction the model neglects
>   (drag test on the table: is `rayleigh_mass_per_s = 0` defensible?).
> - Mass per length and section radius (inertia).
> - Grasp: relative pose of cable and fingers at latch (position/orientation offset,
>   slip under the operating tension) and the translational/rotational compliance of
>   the real mounting; the model's 1e5 N/m and 10 N m/rad encode "no slip", not a
>   measurement. A wrist F/T sensor would also make EI identifiable in the table setup.
> - Independent validation motions (recorded marker trajectories with the gripper
>   poses) to validate the calibrated model on trajectories not used to fit it.

> 2026-09-08: the strain-POD (ModelOrderReduction) chain of
> [sofa_mor_pipeline.md](sofa_mor_pipeline.md) is implemented and gated end to
> end on the SAME SOFA graph: native `WriteState` snapshots, the plugin's POD,
> `ModelOrderReductionMapping`, TS identification on the ROM's own coordinate
> `a`, a visual observer of `a`, and the PDC LMI. Findings that bear on this
> list: r = 16 is the full planar rank (no cheaper model exists at the shape
> tolerance), and the dynamic LMI is again not certified, this time with a
> mechanism: 7-22 uncontrollable eigen-directions on the unit circle per rule
> (PBH), i.e. the weakly excited high strain modes fit as marginal integrators.
> Items 2 and 3 below are therefore the ones that decide the dynamic route.

Items 1–3 of the previous revision are done (model order, boundary block,
gripper in the runtime state) and item 8 is done (the image carries Cosserat
again, and `install_cosserat.sh` repairs a recreated container). The verifier
defect of section 4.9 is fixed and the decisive rerun is done: basic and
relaxed are solver-infeasible on the current model, and the regenerated
subsets (4.6) localise the obstruction to rules 2/3 across the parameter box
and ≥3-rule interactions. Runtime consistency fixes landed with it (the
controller now defaults to `1/Ts` of the identified model instead of 30 Hz,
the camera launch path uses the marker-free detector, the reducer samples TF
at the observation stamp, and `cable_target.yaml` matches the 3-mode state).

1. **Explain rules 2/3.** They are the cells the sound subsets isolate: rules
   0/1 certify across the whole parameter box, 2/3 do not. Check premise-cell
   sample balance, whether the bearing premise's sign convention or the
   excitation's asymmetry concentrates model error there, and run the
   measurements of item 3 on those two rules first.
2. **Redo the controllability audit on the real 8-state model** — PBH plus
   controllability-matrix singular values and a finite-horizon Gramian, per
   parameter vertex, not just a rank.
3. **Attack the rule input spread with real measurements.** Bootstrap
   confidence intervals for every `Aᵢ, Bᵢ` over trajectory resampling,
   principal angles between the `Im(Bᵢ)`, and singular values of each `Bᵢ` —
   `‖ΔB‖/‖B‖` alone is scale-dependent and proves nothing. Then either
   identify the grasp coupling directly from the SOFA scene so `G` is *known*,
   or adopt the position-driven boundary permanently on the strength of 5.14
   plus dedicated persistently-exciting experiments (independent ± pulses and
   multisine per axis around settled equilibria — smooth sweeps leave `g`,
   `ġ`, `q̇` and `u` too correlated to separate `Gu` from `Dq̇`).
4. **Build the basis from training trajectories only** (5.13 leak).
5. **Centre the model on real equilibria and make targets reachable.** Settle
   SOFA at held gripper poses for the equilibrium map `y_eq(g, θ)`; project
   every desired shape onto it (`g* = argmin ‖y_eq(g) − y*‖`, then
   `q* = Φ†(y* − y₀ − Ψg*)`), store the full `[q*, 0, g*]`, verify the
   equilibrium residual, and reject unreachable targets. The current
   controller zeroes the gripper reference, so any non-rest `q*` would chase a
   non-equilibrium — `cable_target.yaml` documents this and pins `q* = 0`.
6. **Match the dataset input channel to the runtime input channel.** The
   dataset applies instantaneous SOFA target jumps + hold; the runtime applies
   a velocity through Servo/robot dynamics. Either interpolate the SOFA
   boundary continuously during substeps (velocity contract), or learn the
   discrete increment map and command increments online (position contract) —
   and for the cascade, identify the outer plant with the inner loop closed or
   bound the interconnection explicitly (separate certificates for the two
   loops do not certify the cascade).
7. Verify the grasp frame in Gazebo (step 2.1) — it biases every identified
   stiffness.
8. Run validation items 3–6 with Gazebo up (sections 6.2–6.3); the camera path
   now exercises `cable_dlo_detector_node` against a real image for the first
   time. While there, make the detector's output honest about uncertainty: it
   currently publishes `confidence = 1.0` with one constant isotropic
   covariance, and the reducer ignores both — interpolated/occluded points
   should carry larger, anisotropic covariance and weigh less in the modal
   projection. Also note what the geometric samples cannot do: they are
   arc-length resamples, not persistent material features, so they cannot
   observe `EA` (the chain is forced to the known length), torsion, or
   material slip along the cable.
9. Optimus is in (3.2). Note the estimator currently only gates on estimate
   freshness — the estimates do not yet update the vertices or gains, so
   "Optimus in the loop" is observation-only until scheduled adaptation is
   designed.
10. Only then wire the cascade (`reference_source:=cable`).

If the corrected relaxed LMI still refuses the dynamic 8-state model, the
defensible fallback is a **quasi-static TS layer**: settle SOFA at a premise
grid, take central-difference shape Jacobians `Jᵢ` per parameter sample,
control the two most reachable shape directions (SVD of `J`) with
`zₖ₊₁ = zₖ + Ts J(ρ,θ)vₖ` and PDC over `Gᵢⱼ = I − Ts Jᵢ Kⱼ` — the same
structure as the working TS-IBVS loop, with the deformation Jacobian in place
of the interaction matrix. Three modes stay for observation; only two
independent static directions exist with two planar inputs, so an arbitrary
3-vector `q*` is not a reachable static target without a third boundary input
(e.g. gripper rotation).

The throwaway diagnostics that produced the numbers in sections 4.6 and
5.11–14 live in `src/cable_ts_control/diag_*.py` (tracked since `b1ea940`;
`diag_lmi.py` carries the corrected acceptance step). They are scratch tools,
not part of the package.

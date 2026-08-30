# Cable TS-PDC shape control — status and diagnosis

Working notes for the SOFA/Gazebo cable identification and control stack.
Written 2026-08-30. Everything below was measured, not estimated; commands to
reproduce each number are given.

---

## 1. Summary

The plant, the observation pipeline and the online parameter identification
work and are verified. The **reduced TS model cannot currently be certified**:
the LMI synthesis is infeasible because the identified local models are not
dissipative. Section 4 is the full diagnosis.

| Plan step | Status |
|---|---|
| 1 — Validate pushed code | **Done.** Both SOFA tests pass |
| 2.1 — Grasp frame | **Not verified.** Needs a visual check in Gazebo |
| 2.2 — Explicit attach state | **Done.** `/cable/attach`, `/cable/detach`, `/cable/grasp_state` |
| 2.3 — Attachment stiffness ≤ 1 mm | **Done.** Measured 0.20 mm |
| 2.4 — Table model | **Done.** Planar constraint (was a silent no-op, see 5.1) |
| 3 — Truth / estimator split | **Done.** Verified non-circular |
| 4.1 — Synthetic observations | **Done** |
| 4.2 — Camera tracker | **Code done, never run against a live camera** |
| 5 — Optimus | **Substituted.** In-process UKF; Optimus itself is *not* installed |
| 6 — Reduced TS model | **Done.** 2 modes, 97.9 % energy |
| 7 — TS vertices from SOFA | **Done.** 16/16 cells identified |
| 8 — PDC synthesis | **BLOCKED.** LMIs infeasible — section 4 |
| 9 — Outer cable TS node | **Done, but Python not C++** (plan asked for `.cpp`) |
| 10 — TS-IBVS integration | **Code done, never run** |
| 11 — Closed-loop launch | **Done.** Parses; never launched end-to-end |
| 12 — Validation order | **Runs 1–2 done, 3–14 not executed** |

Unit tests: **123 passing** (28 identification + 26 perception + 69 control).
`flake8 --select=E,W,F` reports 0 issues.

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

`LogParameterUKF` filters `q = log(θ)`, which is the plan's
`q_EI = log(EI)` / `EI = exp(q_EI)` adapter. Same estimator family as Optimus'
`UKFilterClassic`, so results transfer if the Optimus port lands.

Against an analytic cantilever with 2 mm marker noise, EI recovers
**0.006 → 0.010 within 5 %**. 21 dedicated tests.

Publishes the four topics the plan asks for: `/cable/parameter_estimate`,
`/cable/innovation`, `/cable/marker_rmse`, `/cable/estimator_status`.

### 2.4 TS identification (plan steps 6–7)

```bash
./scripts/run.sh cable_dataset    # 24000 samples, 4 parameter vertices
./scripts/run.sh cable_basis      # 2 modes, 97.91 % energy
./scripts/run.sh cable_identify   # 16/16 cells
```

All 16 cells hold 283–444 training samples; held-out one-step error is
0.33–1.89 mm against a 5 mm threshold. **This looks healthy and is misleading
— see section 4.**

---

## 3. What does not work

### 3.1 LMI synthesis is infeasible — blocks steps 8, 9, 10, 12

```
$ ./scripts/run.sh cable_lmi
synthesising 4 PDC gains for 4 parameter vertices (state 4, input 2)
ERROR: the LMIs are infeasible.
```

Full diagnosis in section 4. This blocks everything downstream:
`cable_ts_model.yaml`, `cable_ts_gains.yaml` and `cable_modal_basis.yaml` are
therefore **not shipped** in `config/`. They are products of the pipeline and
fabricating them would void the stability certificate. Consequently
`cable_closed_loop_sim.launch.py` with `enable_control:=true` cannot start.

### 3.2 Optimus is not installed

The plan's step 5 is not literally satisfied. The Dockerfile has an opt-in
build stage (`--build-arg WITH_OPTIMUS=true`) and a compatibility gate
(`ros2 run cable_identification optimus_smoke_test`), but the gate has **never
passed** because the plugin has never been built. The official binary release
targets SOFA 21.12 while this image is 25.12 — the plan flags this as a real
risk. The in-process UKF covers identification meanwhile.

### 3.3 Cosserat lives only in the running container

Cosserat was installed into the live container, not baked into an image, to
avoid a 7.2 GB rebuild on a disk with 5.6 GB free. **It is lost if the
container is recreated** (`docker compose down`). The Dockerfile stage exists
and is correct; the image simply has not been rebuilt. To restore:

```bash
# re-run the install block, or rebuild the image (needs ~8 GB free):
docker compose build
```

Note `docker image prune` reclaims ~0 B here: the dangling images share all
layers with the tagged one.

### 3.4 Modal reconstruction error exceeds the shape tolerance

The 2-mode basis reconstructs shapes to **14.35 mm RMS**, while
`cable_target.yaml` declares a 5 mm `shape_tolerance_m`. The controller
regulates modal coordinates, so the supervisor's tolerance is applied in modal
space and the two are not directly comparable — but any *marker-space* claim
below ~14 mm is not supported by this basis. Either keep more modes or state
the tolerance in modal terms.

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

---

## 4. Diagnosis: why the LMIs are infeasible

### 4.1 The measurement

Per cell — held-out one-step error, free-running rollout error over 30 steps
(1 s), and the open-loop spectral radius, as reported by `identify_ts_vertices`:

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

**The acceptance gate measured only one-step error.**

Vertex 3, rule 1 is the clearest case: **0.393 mm one-step** — well inside the
5 mm threshold — while diverging to 10¹⁸ mm in free running.

One-step error is nearly blind to eigenvalue error because it predicts
`x(k+1)` from the *true* `x(k)`; error only compounds over a horizon.
Condition numbers are modest (12–50), so this is not raw collinearity. It is
that `fit_local_model` is plain ridge least-squares with **nothing constraining
the fit to be dissipative**, applied to near-unit-root data.

The plan anticipated this: validation item 8 is "TS one-step **and rollout**
prediction test". `rollout_rmse` existed but was never wired into the gate.

### 4.5 What was changed in response

`identify_ts_vertices` now gates on rollout error and spectral radius as well,
so the pipeline fails loudly instead of handing the LMI an unusable model:

```
ERROR: worst rho(A) 4.3809 exceeds 1.0200: a damped cable cannot be
non-dissipative, so the fit is unphysical and no common Lyapunov
function can certify it.
```

New flags: `--max-rollout-rmse` (default 0.05 m), `--rollout-horizon`
(default 30), `--max-spectral-radius` (default 1.02).

This makes the failure honest. **It does not fix it.**

### 4.6 The actual fix (not yet implemented)

Constrain the identification so the fitted `A` is dissipative by construction,
rather than rejecting bad fits afterwards. Options, cheapest first:

1. **Stability-constrained least squares** — solve
   `min ‖X θ − Y‖²  s.t.  ρ(A) ≤ 1` as an SDP, or project each fitted `A` onto
   the Schur-stable set. Localised change to `fit_local_model`.
2. **Physically structured `A`** — the state is `[q, q̇]`, so impose the
   second-order form `A = [[I, dt·I], [−dt·K, I − dt·D]]` with `K, D ⪰ 0`.
   Far fewer free parameters and dissipativity is automatic.
3. **Longer prediction horizon in the loss** — fit on multi-step error instead
   of one-step, which penalises eigenvalue error directly.

Option 2 is the most principled: it encodes what the cable actually is.

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
reconstruction error improved 15.25 → 6.21 mm.

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

---

## 6. Reproducing

```bash
# plant
./scripts/run.sh cable_plugin_test
./scripts/run.sh cable_forward_test

# identification pipeline
./scripts/run.sh cable_dataset
./scripts/run.sh cable_basis
./scripts/run.sh cable_identify     # currently fails the rho(A) gate
./scripts/run.sh cable_lmi          # infeasible

# unit tests
./scripts/run.sh cable_unit_tests

# identification milestone (needs Gazebo)
./scripts/run.sh cable_identify_sim
```

Build on a memory-constrained host — the default 16-way parallelism OOMs at
8 GB:

```bash
MAKEFLAGS=-j2 colcon build --executor sequential --parallel-workers 1
```

`ament_python` installs by copy, not symlink: rebuild the package before
running pytest or the stale installed copy is tested.

---

## 7. Next steps, in order

1. Implement stability-constrained `fit_local_model` (section 4.6). Nothing
   downstream can be certified until this lands.
2. Re-run identify → LMI; confirm `worst |eig(A−BK)| < 1`.
3. Verify the grasp frame in Gazebo (step 2.1) — it biases every stiffness.
4. Run validation items 3–5 with Gazebo up.
5. Run item 6, EI recovery in the closed loop, against the live estimator.
6. Rebuild the image so Cosserat survives container recreation; optionally
   with `WITH_OPTIMUS=true` and run the compatibility gate.
7. Only then wire the cascade (`reference_source:=cable`).

# Cable TS-PDC shape control — status and diagnosis

Working notes for the SOFA/Gazebo cable identification and control stack.
Written 2026-08-30. Everything below was measured, not estimated; commands to
reproduce each number are given.

---

## 1. Summary

The plant, the observation pipeline and the online parameter identification
work and are verified. The **reduced TS model cannot yet be certified**: the
LMI synthesis is infeasible because the four local rule models drift too far
apart to share one quadratic Lyapunov function. Section 4 is the full
diagnosis, including two retracted claims from the first version of this
document.

Current model: state `[q₁ q₂ q̇₁ q̇₂ p_gx p_gy]` (6), input `[v_x v_y]` (2),
4 rules, `Ts = 0.04 s` (25 Hz). Envelope `EI ∈ [0.005, 0.015]`,
`rayleigh_stiffness ∈ [0.018, 0.022]`.

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
| 6 — Reduced TS model | **Done.** 2 modes, 95.1 % energy, 13.4 mm reconstruction |
| 7 — TS vertices from SOFA | **Done.** 16/16 cells, held-out one-step 3.9–4.5 mm |
| 8 — PDC synthesis | **BLOCKED.** LMIs infeasible — section 4 |
| 9 — Outer cable TS node | **Done, but Python not C++** (plan asked for `.cpp`) |
| 10 — TS-IBVS integration | **Code done, never run** |
| 11 — Closed-loop launch | **Done.** Parses; never launched end-to-end |
| 12 — Validation order | **Runs 1–2 done, 3–14 not executed** |

Unit tests: **148 passing** (28 identification + 26 perception + 94 control).
`flake8 --select=E,W,F` reports 0 issues. Head is `da6d7da`.

**Nothing in the control loop can run today**: no gains exist (section 3.1) and
the runtime state does not match the identified model (section 3.7).

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
./scripts/run.sh cable_dataset    # 8000 samples: 4 vertices x 5 trajectories
./scripts/run.sh cable_basis      # 2 modes, 95.12 % energy, 13.4 mm rmse
./scripts/run.sh cable_identify   # 16/16 cells
```

The dataset is sampled at the exact controller period (40 ms = 4 SOFA
substeps, zero-order hold) and split into train/validation by whole
trajectory, so the rollout below is a real contiguous prediction rather than a
walk over scattered samples.

| parameter vertex | held-out one-step | blended rollout (2 s) | max ρ(A) |
|---|---|---|---|
| EI 0.005, damping 0.018 | 4.14 mm | 289 mm | 1.0000 |
| EI 0.005, damping 0.022 | 3.93 mm | 123 mm | 1.0000 |
| EI 0.015, damping 0.018 | 4.22 mm | 194 mm | 1.0000 |
| EI 0.015, damping 0.022 | 4.48 mm | 208 mm | 1.0000 |

`ρ(A) = 1.0000` is exactly right: the gripper integrator sits on the unit
circle and the cable modes are strictly inside. The free-running error is still
far above the 50 mm gate, so `cable_identify` only writes a model when that
gate is relaxed explicitly.

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

Full diagnosis in section 4. This blocks everything downstream:
`cable_modal_basis.yaml`, `cable_ts_model.yaml` and `cable_ts_gains.yaml` are
therefore **not shipped** in `config/` — only `cable_parameter_bounds.yaml` and
`cable_target.yaml` are. The others are products of the pipeline and
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

The 2-mode basis reconstructs shapes to **13.4 mm RMS**, while
`cable_target.yaml` declares a 5 mm `shape_tolerance_m`. The controller
regulates modal coordinates, so the supervisor's tolerance is applied in modal
space and the two are not directly comparable — but any *marker-space* claim
below ~13 mm is not supported by this basis.

This is very likely the real accuracy floor: the held-out one-step error sits
at 3.9–4.5 mm and did **not** move for any change to the fit (joint fuzzy
regression, gripper in the state, structured second-order form, passivity
projection, coherence). A fit cannot beat its own representation.

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

### 3.7 The runtime state does not match the identified model

Adding the gripper position to the state (section 4.5) changed the model but
not the node that feeds it:

| producer | contents | size |
|---|---|---|
| `cable_state_reducer_node` | `[q, q̇]` | 4 |
| identified `cable_ts_model.yaml` | `[q, q̇, p_g]` | 6 |

`cable_ts_controller_node` now checks this and refuses to run rather than
feeding a 4-vector into a 6-state model:

```
reduced state has 4 entries but the model expects 6; refusing to control
```

The same change also exposed `n_modes = state_dim // 2`, which returns 3 for a
2-mode model once the gripper is in the state and silently mis-sizes every
shape target. `n_modes` is now stored explicitly in the model and covered by
tests. **The mismatch itself is not fixed**: the reducer must publish the
gripper position from TF before any gains could run.

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
| `worst_spectral_radius` on averaged cross terms only | stops rejecting valid certificates |

A new finding along the way: **the joint fuzzy regressor is structurally rank
deficient**. Complementary triangular memberships are exactly affine in the
premises (`h₂ + h₃ = (x₁ − lo)/(hi − lo)` to 1e-16), so wherever a premise is
also a state component the `hᵢx` blocks are linearly dependent — σ_min ≈ 1e-15
with and without an affine term. The blended prediction is unique; the vertex
matrices are not. The fit therefore takes the minimum-norm solution.

### 4.6 Where it stands now

With the structured fit, every `A_i` has `ρ(A) = 1.0000` exactly — the gripper
integrator, with the cable modes strictly inside. That is physically correct
and was not true before.

On the current envelope (`EI ∈ [0.005, 0.015]`, damping `∈ [0.018, 0.022]`):

| subset | diagonal blocks only | with rule-transition terms |
|---|---|---|
| each single `(vertex, rule)` | yes, all 16 | — |
| vertex 0 (4 rules) | yes | **yes** |
| vertex 1 (4 rules) | yes | **yes** |
| vertex 2 (4 rules) | yes | **yes** |
| vertex 3 (4 rules) | yes | no |
| all four vertices | no | no |

Every rule *pair* inside a vertex is also feasible. Before these corrections no
subset beyond a single rule was feasible at all, so three of four parameter
vertices are now individually certifiable — but the shared certificate across
the whole box still is not.

### 4.7 What is still open

**The LMI implementation is not the problem.** It was checked line by line
against Tanaka & Wang (2001): the basic conditions match (3.19)–(3.20) and the
relaxed ones match (3.27)–(3.28). Taking the Schur complement of the
implemented blocks returns `G'PG − P + (s−1)PYP < 0` and `H'PH − P − PYP ≤ 0`,
i.e. exactly Theorem 10 with `Q = PYP`.

Three hypotheses were tested:

| Hypothesis | Result |
|---|---|
| Bad state scaling | **no** — three similarity transforms, all still infeasible |
| Not enough gain freedom | **no** — parameters as premises (16 rules, 16 gains) still infeasible |
| Rules too dissimilar | **yes** |

Within a *single* parameter vertex the four rules differ by
`max‖Aᵢ−Aⱼ‖ = 1.92` against `‖A‖ ≈ 2.5`, and `max‖Bᵢ−Bⱼ‖ = 2.05` against
`‖B‖ ≈ 1.0`. Local linearisations of one smooth plant at neighbouring corners
of the premise box should be far closer than that.

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
triples. The one-step error is flat (4.36–4.48 mm) across the whole sweep, so
the penalty buys certifiability without buying accuracy. It is off by default
for that reason: a diagnostic, not a solution.

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

Confirm: UR5e visible, cable visible, seven distinctly coloured marker spheres,
fixture static, cable follows the gripper after attachment. This is also where
step 2.1 (the grasp frame) finally gets checked.

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
./scripts/run.sh cable_lmi          # infeasible; writes nothing
```

### 6.5 Unit tests

```bash
./scripts/run.sh cable_unit_tests   # 148 tests
```

### 6.6 What cannot be run

```bash
./scripts/run.sh cable_closed_loop enable_control:=true   # no gains exist
```

Blocked twice over: `cable_ts_gains.yaml` is never produced (section 3.1), and
even with gains the controller would refuse on the state mismatch
(section 3.7). Validation items 7–14 are therefore unreachable today.

### 6.7 Build notes

The default 16-way parallelism OOMs on an 8 GB host:

```bash
MAKEFLAGS=-j2 colcon build --executor sequential --parallel-workers 1
```

`ament_python` installs by copy, not symlink: rebuild the package before
running pytest, or the stale installed copy is tested.

---

## 7. Next steps, in order

1. **Choose the mode count on held-out marker error.** The one-step error is
   pinned at ~4.4 mm through every change to the fit, while the basis
   reconstructs to 13.4 mm. The representation, not the identification, is the
   binding constraint — and no amount of LMI work fixes that.
2. **Build the basis from training trajectories only**, so the held-out numbers
   mean what they say.
3. **Centre the model on a real equilibrium.** Settle SOFA at a held gripper
   pose to get `(q*, p_g*)`; the regulator LMIs assume the origin is an
   equilibrium and `q* = 0` is only the PCA mean of a dynamic dataset.
4. Re-run identify → LMI and check `max eig(G'PG − P) < 0` *and*
   `worst |eig(A − BK)| < 1`. Never trust the solver's own status alone
   (section 4.8).
5. **Publish the gripper position from the state reducer** so the runtime state
   matches the model (section 3.7).
6. Verify the grasp frame in Gazebo (step 2.1) — it biases every identified
   stiffness.
7. Run validation items 3–6 with Gazebo up (section 6.2–6.3).
8. Rebuild the image so Cosserat survives container recreation; optionally with
   `WITH_OPTIMUS=true`, then run the compatibility gate.
9. Only then wire the cascade (`reference_source:=cable`).

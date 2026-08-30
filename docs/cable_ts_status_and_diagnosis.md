# Cable TS-PDC shape control — status and diagnosis

Working notes for the SOFA/Gazebo cable identification and control stack.
Written 2026-08-30. Everything below was measured, not estimated; commands to
reproduce each number are given.

---

## 1. Summary

The plant, the observation pipeline and the online parameter identification
work and are verified. The reduced model is now good enough to be worth
certifying — the free-running rollout is inside the 50 mm gate on a matched
dataset for the first time — but the **PDC synthesis is still infeasible**.
Section 4 is the full diagnosis.

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
| `corr(q, gripper)` | 0.76 | **0.00** |
| 4-vertex LMI | infeasible | infeasible |

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
| 5 — Optimus | **Substituted.** In-process UKF; Optimus itself is *not* installed |
| 6 — Reduced model | **Done.** Craig-Bampton split, 3 modes, 3.50 mm |
| 7 — TS vertices from SOFA | **Done.** 16/16 cells, held-out one-step 3.76 mm |
| 8 — PDC synthesis | **BLOCKED.** LMIs infeasible — section 4 |
| 9 — Outer cable TS node | **Done, but Python not C++** (plan asked for `.cpp`) |
| 10 — TS-IBVS integration | **Code done, never run** |
| 11 — Closed-loop launch | **Done.** Parses; never launched end-to-end |
| 12 — Validation order | **Runs 1–2 done, 3–14 not executed** |

Unit tests: **190 passing** (28 identification + 48 perception + 114 control).

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

`LogParameterUKF` filters `q = log(θ)`, which is the plan's
`q_EI = log(EI)` / `EI = exp(q_EI)` adapter. Same estimator family as Optimus'
`UKFilterClassic`, so results transfer if the Optimus port lands.

Against an analytic cantilever with 2 mm marker noise, EI recovers
**0.006 → 0.010 within 5 %**. 21 dedicated tests.

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
to the 5 mm shape tolerance (see 5.12).

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

The 3-mode basis reconstructs shapes to **3.50 mm** against the 5 mm
`shape_tolerance_m` in `cable_target.yaml`. The previous 2-mode plain-PCA basis
sat at 13.4 mm, which was the real accuracy floor of the whole pipeline. The
improvement is the Craig-Bampton split (5.13), not a bigger basis: at the same
order the split alone takes 8.87 mm to 3.50 mm.

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
| `worst_spectral_radius` on averaged cross terms only | stops rejecting valid certificates |

A new finding along the way: **the joint fuzzy regressor is structurally rank
deficient**. Complementary triangular memberships are exactly affine in the
premises (`h₂ + h₃ = (x₁ − lo)/(hi − lo)` to 1e-16), so wherever a premise is
also a state component the `hᵢx` blocks are linearly dependent — σ_min ≈ 1e-15
with and without an affine term. The blended prediction is unique; the vertex
matrices are not. The fit therefore takes the minimum-norm solution.

### 4.6 Where it stands now (2026-08-30, after the four fixes of section 5)

Every `A_i` has `ρ(A) = 1.0000` exactly — the gripper integrator, with the
cable modes strictly inside. Every `(A_i, B_i)` pair is fully controllable
(rank 6/6 at 2 modes, checked by the PBH test at every eigenvalue on or outside
the unit circle), so each is individually stabilisable and the obstruction is
purely the *common* Lyapunov matrix.

Minimal infeasible subsets on the 3-mode Craig-Bampton model
(`src/cable_ts_control/diag_lmi.py`):

| subset | certified |
|---|---|
| one parameter vertex, 4 rules, cross terms | no, all four |
| one rule, all four parameter vertices | rule 0 only |
| rules 0+1, 0+3, 1+2 inside vertex 0 | yes |
| rules 0+2, 1+3, 2+3 inside vertex 0 | no |

Four candidate causes were tested and **eliminated by measurement**:

| candidate | test | result |
|---|---|---|
| parameter box too wide | `EI ∈ [0.008, 0.012]`, ±20 % instead of 3:1 | still infeasible |
| premise box too wide | `--premise-margin` 1.0 → 0.3 | still infeasible at every width |
| velocity-filter lag | `--velocity-alpha` 0.4 → 1.0 | still infeasible |
| decay margin too tight | `decay` 0.999 → 0.95 | still infeasible |

What is left is the spread of the rule **input** matrices:
`max‖Bᵢ−Bⱼ‖ / ‖B‖ = 1.6 … 3.3`. The rules disagree about the input gain by
more than the gain itself, and no single `K_j` can serve them all. Section 5.14
identifies the mechanism (the `G u` column is nearly collinear with the
`D q̇` column, so each rule splits them differently) and shows that removing
that term costs nothing on held-out data. Doing so moves the 3-mode problem
from *infeasible* to *the solver returns a point that fails verification* —
progress, but not a certificate, and the script correctly refuses to write it.

### 4.7 What is still open

**The LMI implementation is not the problem.** It was checked line by line
against Tanaka & Wang (2001): the basic conditions match (3.19)–(3.20) and the
relaxed ones match (3.27)–(3.28). Taking the Schur complement of the
implemented blocks returns `G'PG − P + (s−1)PYP < 0` and `H'PH − P − PYP ≤ 0`,
i.e. exactly Theorem 10 with `Q = PYP`.

Hypotheses tested, all on the corrected model:

| Hypothesis | Result |
|---|---|
| Bad state scaling | **no** — three similarity transforms, all still infeasible |
| Not enough gain freedom | **no** — parameters as premises (16 rules, 16 gains) still infeasible |
| Uncontrollable mode | **no** — every vertex is rank 6/6 controllable |
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

Replaced with the standard Craig-Bampton split into a constraint mode and
fixed-interface normal modes:

```
y = y0 + Psi g + Phi q
```

`Psi g` is the quasi-static response to the prescribed boundary; `q` describes
only what the boundary does not explain, and is orthogonal to `g` by
construction. At equal model order:

| | plain PCA | Craig-Bampton |
|---|---|---|
| reconstruction, 3 modes | 8.87 mm | **3.50 mm** |
| `corr(q, gripper)` | 0.7577 | **0.0000** |
| held-out one-step | 6.56 mm | **3.80 mm** |
| held-out rollout | 95.3 mm | **38.6 mm** |

`ModalBasis` carries the optional `psi` block and the `boundary_reference` it
is measured from; `psi = None` reproduces the old plain PCA exactly.

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
want spheres at (it defaults to empty), and run
`cable_marker_tracker_node` instead of `cable_dlo_detector_node`.

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

Items 1–3 of the previous revision are done (model order, boundary block,
gripper in the runtime state) and item 8 is done (the image carries Cosserat
again, and `install_cosserat.sh` repairs a recreated container).

1. **Attack the rule input spread.** `‖ΔB‖/‖B‖ = 1.6–3.3` is the last measured
   obstruction (section 4.6). Two things to try, in this order: identify the
   grasp coupling's damping directly from the SOFA scene rather than regressing
   it, so `G` is *known* instead of fitted; and, failing that, adopt the
   position-driven boundary permanently (`--no-input-feedthrough`) on the
   strength of the held-out evidence in 5.14. Nothing else moved the
   feasibility.
2. **Build the basis from training trajectories only.** It is currently built
   from all of them, so the held-out numbers are mildly optimistic.
3. **Centre the model on a real equilibrium.** Settle SOFA at a held gripper
   pose to get `(q*, p_g*)`; the regulator LMIs assume the origin is an
   equilibrium and `q* = 0` is only the mean of a dynamic dataset.
4. Re-run identify → LMI and check `max eig(G'PG − P) < 0` *and*
   `worst |eig(A − BK)| < 1`. Never trust the solver's own status alone
   (section 4.8) — it reported *feasible* on a point with `ρ = 1.25` during
   this work.
5. Verify the grasp frame in Gazebo (step 2.1) — it biases every identified
   stiffness.
6. Run validation items 3–6 with Gazebo up (sections 6.2–6.3), and point the
   camera path at `cable_dlo_detector_node` so the marker-free observation is
   exercised against a real image for the first time.
7. Optionally rebuild with `WITH_OPTIMUS=true` and run the compatibility gate.
8. Only then wire the cascade (`reference_source:=cable`).

The throwaway diagnostics that produced the numbers in sections 4.6 and 5.11–14
are left in `src/cable_ts_control/diag_*.py`. They are untracked scratch, not
part of the package.

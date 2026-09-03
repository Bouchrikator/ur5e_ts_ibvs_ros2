"""Optimus gates on the Cosserat cable (protocol optimus_sofa25_portage.md, §11).

Headless, no ROS. Decides whether the ported Optimus identifies the cable
stiffnesses through SOFA:

  D  parameter transform: OptimParams log/exp round trip, initial log-variance
     equals the log-normal formula, invalid initial values rejected.
  E  Cosserat sensitivity: a candidate EI (GJ) written through OptimParams
     rebuilds the section stiffness (force ratio EI2/EI1, §7.4), changes the
     prediction, and the first correction moves the parameter (P_theta_y != 0).
  H  boundary conditions: every sigma point of a filter step sees the same
     gripper pose (one BoundaryController application per propagation, §8.2).
  F  synthetic EI recovery: truth 0.010 N.m^2, start 0.006, marker noise 2 mm;
     < 5 % final error, finite positive shrinking variance, innovation down on
     the informative window, held-out trajectory. Run twice: an observation at
     every step, and one every 3 steps (prediction-only steps in between).
  G  GJ recovery under a torsional excitation (truth 0.008, start 0.012).

    ros2 run cable_identification optimus_recovery_test [--steps N] [--seed S]
"""

import argparse
import math
import os
import sys

import numpy as np

TRUTH = {"EI": 0.010, "GJ": 0.008}     # cable_truth.yaml
START = {"EI": 0.006, "GJ": 0.012}     # cable_estimator_initial.yaml
OBS_STD_M = 0.002                       # protocol Gate F
RELATIVE_STD = 0.30                     # 30 % initial relative uncertainty (§6.3)
IDENTITY_Q = [0.0, 0.0, 0.0, 1.0]


def experiment_cfg():
    """Hanging cable under gravity, base driven by the gripper, free tip.
    Geometry and true parameters from cable_truth.yaml."""
    from cable_identification import cosserat_model as cm

    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("cable_identification")
        cfg = cm.load_config(os.path.join(share, "config", "cable_truth.yaml"))
    except Exception:
        cfg = cm.load_config()
        cfg.update(length_m=0.7, mass_kg=0.07, number_of_sections=16, number_of_frames=40)
    cfg.update(EI_Nm2=TRUTH["EI"], GJ_Nm2=TRUTH["GJ"], gravity=[0.0, 0.0, -9.81],
               planar=False, grasp_tip=False)
    return cfg


def _quat_x(angle):
    return [math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)]


def flexion_pose(t):
    """EI experiment: base swings 5 cm in y at 0.5 Hz and 3 cm in z at 0.3 Hz."""
    return [0.0, 0.05 * math.sin(2 * math.pi * 0.5 * t),
            0.03 * math.sin(2 * math.pi * 0.3 * t)] + IDENTITY_Q


def torsion_pose(t):
    """GJ experiment: base rolls +-20 deg about the rod axis at 0.4 Hz. Gravity keeps
    the sag plane vertical, torsion drags it along: how far the sag plane follows
    the base is the GJ signature (GJ 0.012 vs 0.008: 20 mm rms, 10 sigma_obs).
    Larger rolls are more sensitive but the first corrections then overshoot and
    leave a residual state error in the weakly damped twist modes."""
    return [0.0, 0.0, 0.0] + _quat_x(math.radians(20.0) * math.sin(2 * math.pi * 0.4 * t))


def validation_pose(t):
    """Held out: slower, larger, different phase, plus a mild roll."""
    return [0.02 * math.sin(2 * math.pi * 0.2 * t), 0.08 * math.cos(2 * math.pi * 0.35 * t),
            0.0] + _quat_x(math.radians(20.0) * math.sin(2 * math.pi * 0.25 * t))


def build_truth(cfg, values):
    import Sofa.Core
    import Sofa.Simulation
    from cable_identification import cosserat_model as cm

    c = dict(cfg)
    c["EI_Nm2"] = float(values.get("EI", cfg["EI_Nm2"]))
    c["GJ_Nm2"] = float(values.get("GJ", cfg["GJ_Nm2"]))
    root = Sofa.Core.Node("truth")
    cm.prepare_root(root, c)
    cable = cm.build_cable(root, c)
    Sofa.Simulation.init(root)
    return root, cable


def build_estimator(cfg, parameters, start):
    import Sofa.Core
    import Sofa.Simulation
    from cable_identification import cosserat_model as cm
    from cable_identification import optimus_scene as osc

    root = Sofa.Core.Node("estimator")
    cm.prepare_root(root, cfg, animation_loop=None)
    est = osc.build_optimus_cable(root, cfg, parameters=parameters,
                                  init_values={p: start[p] for p in parameters},
                                  relative_std=RELATIVE_STD, observation_std=OBS_STD_M)
    Sofa.Simulation.init(root)
    return root, est


def drive(cable, root, pose):
    """Kinematic boundary for the truth plant. `set_base_pose` alone is applied one
    solve late (the mapped frames the forces use are refreshed only after the solve);
    the estimator's sigma points always start from the current boundary, so the
    reference plant gets the same semantics (verified identical to 0.00 mm with the
    parameter at its true value)."""
    del root
    cable.set_base_pose(pose)
    cable.refresh_mapping()


def rollout(cfg, values, pose_fn, steps):
    """Marker trajectories of a cable with the given stiffnesses along `pose_fn`."""
    import Sofa.Simulation

    root, cable = build_truth(cfg, values)
    dt = root.dt.value
    out = []
    for k in range(steps):
        drive(cable, root, pose_fn(k * dt))
        Sofa.Simulation.animate(root, dt)
        out.append(np.array(cable.marker_positions()))
    return np.array(out)


def marker_rmse(a, b):
    """RMS Euclidean error per marker over all samples [m]."""
    d = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=-1))))


def gate_e_force_ratio(cfg, parameter):
    """§7.4: same pure strain state, theta2 = 2 theta1 -> internal force ratio 2 and a
    different one-step prediction; back to theta1 reproduces the first prediction."""
    import Sofa.Core
    import Sofa.Simulation
    from cable_identification import cosserat_model as cm
    from cable_identification import optimus_scene as osc

    field, component = {"EI": ("EI", 2), "GJ": ("GI", 0)}[parameter]
    c = dict(cfg)
    c["gravity"] = [0.0, 0.0, 0.0]
    root = Sofa.Core.Node("sensitivity")
    root.addObject("RequiredPlugin", name="optimus_plugin", pluginName=["Optimus"])
    cm.prepare_root(root, c)
    cable = cm.build_cable(root, c)
    param = cable.solver_node.addObject(
        "OptimParams", name=f"estimated_{parameter}", template="Vector", optimize=True,
        numParams=1, initValue=[TRUTH[parameter]], stdev=[osc.optimus_stdev(RELATIVE_STD)],
        transformParams="exponential")
    cable.force_field.findData(field).setParent(param.findData("value"))
    Sofa.Simulation.init(root)

    strain = np.zeros_like(cable.strain_mo.position.value)
    strain[:, component] = 2.0   # pure bending (about z) or pure torsion, 2 rad/m
    with cable.strain_mo.reset_position.writeable() as p:
        p[:] = strain

    def step_with(theta):
        # Simulation.reset: identical independent state (reset_position, zero velocity)
        # AND freshly propagated mappings for every propagation. Writing positions
        # directly would leave the Cosserat mapping cache from the previous step.
        Sofa.Simulation.reset(root)
        param.findData("value").value = [float(theta)]
        Sofa.Simulation.animate(root, root.dt.value)   # DefaultAnimationLoop runs UpdateInternalData
        force = np.array(cable.strain_mo.force.value)
        # full poses: torsion of a straight rod moves orientations, not positions
        return float(np.linalg.norm(force[:, component])), np.array(cable.frame_poses())

    f1, frames1 = step_with(TRUTH[parameter])
    f2, frames2 = step_with(2.0 * TRUTH[parameter])
    f1b, frames1b = step_with(TRUTH[parameter])
    return {
        "read_back": float(cable.force_field.findData(field).value),
        "force_ratio": f2 / f1 if f1 > 0 else float("nan"),
        "prediction_moved": float(np.max(np.abs(frames2 - frames1))),
        "restored": float(np.max(np.abs(frames1b - frames1))),
    }


def run_filter(cfg, parameters, pose_fn, steps, rng, obs_every=1):
    """Truth plant + Optimus estimator in lock-step. Returns per-step traces."""
    import Sofa.Simulation

    truth_root, truth = build_truth(cfg, TRUTH)
    est_root, est = build_estimator(cfg, parameters, START)
    dt = truth_root.dt.value
    n_sigma = len(parameters) + 1   # simplex sigma points

    init_state = np.array(est.roukf.state.value, dtype=float)
    init_var = est.log_variances()
    est_trace = {p: [] for p in parameters}
    innovations, variances, tracking = [], [], []
    for k in range(steps):
        pose = pose_fn(k * dt)
        drive(truth, truth_root, pose)
        Sofa.Simulation.animate(truth_root, dt)
        clean = np.array(truth.marker_positions())
        obs = clean + rng.normal(0.0, OBS_STD_M, (7, 3))

        # u[k] drives k -> k+1 for both plants; the observation is y[k+1] (§8.3)
        est.set_boundary_pose(pose)
        est.set_observation(obs, valid=(k % obs_every == 0))
        before = est.controller.applications
        Sofa.Simulation.animate(est_root, dt)
        if est.controller.applications - before != n_sigma:
            raise AssertionError(
                f"step {k}: {est.controller.applications - before} boundary applications, "
                f"expected {n_sigma} (one per sigma point)")

        for p, v in est.estimates().items():
            est_trace[p].append(v)
        innovations.append(est.innovation_rms())
        variances.append(est.log_variance(parameters[0]))
        tracking.append(marker_rmse(est.cable.marker_positions(), clean))
    return {
        "est": {p: np.array(v) for p, v in est_trace.items()},
        "innovation": np.array(innovations),
        "variance": np.array(variances),
        "tracking": np.array(tracking),
        "init_state": init_state,
        "init_var": init_var,
        "final_state": np.array(est.roukf.state.value, dtype=float),
        "final": est.estimates(),
    }


def convergence_step(trace, truth, tol=0.05):
    err = np.abs(trace - truth) / truth
    idx = np.nonzero(err < tol)[0]
    return int(idx[0]) + 1 if len(idx) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=300, help="filter steps (dt from config)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import SofaRuntime  # noqa: F401  (SOFA_ROOT plugin search paths)
    from cable_identification import optimus_scene as osc

    cfg = experiment_cfg()
    rng = np.random.default_rng(args.seed)
    steps = args.steps
    results = []

    def check(name, ok, detail):
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    def recovery_gates(tag, parameter, run):
        truth, start = TRUTH[parameter], START[parameter]
        trace = run["est"][parameter]
        final = run["final"][parameter]
        rel_err = abs(final - truth) / truth
        n_conv = convergence_step(trace, truth)
        marks = [1, 2, 5, 10, 25, 50, 100, 200, steps]
        print(f"      {parameter} trace: " + ", ".join(
            f"k={m}:{trace[m - 1]:.5f}" for m in marks if m <= steps))
        q = max(1, steps // 4)
        quarters = [run["tracking"][i * q:(i + 1) * q] for i in range(4)]
        print("      state tracking rmse (est markers vs clean truth) per quarter [mm]: "
              + ", ".join(f"{1e3 * np.mean(x):.2f}" for x in quarters if len(x)))
        print("      innovation rms per quarter [mm]: " + ", ".join(
            f"{1e3 * np.nanmean(run['innovation'][i * q:(i + 1) * q]):.2f}" for i in range(4)))
        early = trace[:min(50, steps)]
        check(f"{tag}: {parameter} moves within the first 50 corrections (P_theta_y != 0)",
              np.any(np.abs(early - start) > 1e-9 * start),
              f"max |{parameter} - start| over steps 1-50 = {np.max(np.abs(early - start)):.3e}")
        check(f"{tag}: final relative error on {parameter} < 5 %", rel_err < 0.05,
              f"{parameter}={final:.6f} (truth {truth}), error {100 * rel_err:.2f} %, "
              f"converged at step {n_conv}")
        var = run["variance"]
        check(f"{tag}: log-variance finite, positive and reduced",
              np.all(np.isfinite(var)) and var[-1] > 0 and var[-1] < var[0],
              f"{var[0]:.3e} -> {var[-1]:.3e} (log space)")
        innov = run["innovation"]
        floor = OBS_STD_M * math.sqrt(3)   # rms per marker of 3-axis white noise
        pre = float(np.nanmean(innov[:max(1, (n_conv or steps) - 1)]))
        post = float(np.nanmean(innov[-max(1, steps // 4):]))
        check(f"{tag}: innovation down to the noise floor after convergence",
              post <= max(pre, 1.2 * floor),
              f"before convergence {1e3 * pre:.2f} mm -> last quarter {1e3 * post:.2f} mm "
              f"(noise floor {1e3 * floor:.2f} mm)")
        val_true = rollout(cfg, TRUTH, validation_pose, steps)
        val_est = rollout(cfg, {**TRUTH, parameter: final}, validation_pose, steps)
        val_start = rollout(cfg, {**TRUTH, parameter: start}, validation_pose, steps)
        val_tol = rollout(cfg, {**TRUTH, parameter: 1.05 * truth}, validation_pose, steps)
        rmse_est, rmse_start = marker_rmse(val_true, val_est), marker_rmse(val_true, val_start)
        rmse_tol = marker_rmse(val_true, val_tol)
        check(f"{tag}: held-out trajectory: better than start and within the 5 % parameter tolerance",
              np.isfinite(rmse_est) and rmse_est < rmse_start and rmse_est <= max(OBS_STD_M, rmse_tol),
              f"validation_pose rmse({parameter}_est)={1e3 * rmse_est:.2f} mm, rmse({parameter}_start)="
              f"{1e3 * rmse_start:.2f} mm, rmse({parameter} +5 %)={1e3 * rmse_tol:.2f} mm")


    print("=" * 70)
    print("OPTIMUS RECOVERY GATES (D, E, H, F, G)")
    print("=" * 70)

    # --- Gate D: log transform and initial covariance (protocol §6.3 numbers)
    s_opt = osc.optimus_stdev(RELATIVE_STD)
    check("D: stdev for OptimParams = exp(sqrt(log(1+c^2)))", abs(s_opt - 1.341194157207) < 1e-9,
          f"c=0.30 -> stdev={s_opt:.12f}")
    check("D: OptimParams initial log-variance = log(1+c^2)",
          abs(osc.initial_log_variance(s_opt) - math.log1p(RELATIVE_STD ** 2)) < 1e-12,
          f"{osc.initial_log_variance(s_opt):.12f}")
    rejected = []
    for bad in (0.0, -0.006, float("nan"), float("inf")):
        try:
            import Sofa.Core
            osc.build_optimus_cable(Sofa.Core.Node("bad"), cfg, parameters=("EI",),
                                    init_values={"EI": bad})
        except ValueError:
            rejected.append(bad)
    check("D: zero/negative/non-finite initial values rejected", len(rejected) == 4,
          f"rejected {rejected}")

    # --- Gate E: Cosserat sensitivity through OptimParams -> Data link -> tracker
    for parameter, label in (("EI", "bending"), ("GJ", "torsion")):
        e = gate_e_force_ratio(cfg, parameter)
        check(f"E: {parameter} link read back by the force field",
              abs(e["read_back"] - TRUTH[parameter]) < 1e-12, f"{parameter}={e['read_back']}")
        check(f"E: {label} force scales with {parameter} (ratio 2.0)",
              abs(e["force_ratio"] - 2.0) < 1e-6, f"ratio={e['force_ratio']:.9f}")
        check(f"E: one-step prediction depends on {parameter}", e["prediction_moved"] > 1e-9,
              f"max pose component shift={e['prediction_moved']:.3e}")
        check(f"E: back to {parameter}1 restores the prediction",
              e["restored"] < 1e-3 * e["prediction_moved"],
              f"max diff={e['restored']:.3e}")

    for parameter, pose_fn in (("EI", flexion_pose), ("GJ", torsion_pose)):
        sep = marker_rmse(rollout(cfg, TRUTH, pose_fn, steps),
                          rollout(cfg, {**TRUTH, parameter: START[parameter]}, pose_fn, steps))
        check(f"E: {parameter} {START[parameter]} vs {TRUTH[parameter]} separates the markers "
              f"(> 3 sigma_obs)", sep > 3 * OBS_STD_M, f"rmse={1e3 * sep:.2f} mm")

    # --- Gates D (in situ), H, F: EI, observation at every step
    run = run_filter(cfg, ("EI",), flexion_pose, steps, rng)
    check("D: ROUKF initial state is log(EI0)", abs(run["init_state"][-1] - math.log(START["EI"])) < 1e-9,
          f"q0={run['init_state'][-1]:.12f} (log 0.006 = -5.115995809754)")
    check("D: ROUKF initial reduced variance = log(1+c^2)",
          abs(run["init_var"]["EI"] - math.log1p(RELATIVE_STD ** 2)) < 1e-9,
          f"{run['init_var']['EI']:.9f}")
    check("D: physical value = exp(log state) after the run",
          abs(math.exp(run["final_state"][-1]) - run["final"]["EI"]) < 1e-12 * run["final"]["EI"],
          f"exp(q)={math.exp(run['final_state'][-1]):.9f}, value={run['final']['EI']:.9f}")
    check("H: one boundary application per sigma point", True, f"{steps} steps x 2")
    recovery_gates("F", "EI", run)

    # --- Gate F bis: observation every 3rd step (prediction-only steps, §8.4)
    run3 = run_filter(cfg, ("EI",), flexion_pose, steps, rng, obs_every=3)
    recovery_gates("F/3", "EI", run3)

    # --- Gate G: GJ under torsional excitation
    rung = run_filter(cfg, ("GJ",), torsion_pose, steps, rng)
    recovery_gates("G", "GJ", rung)

    ok = all(r[1] for r in results)
    print("OPTIMUS_RECOVERY_TEST_" + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

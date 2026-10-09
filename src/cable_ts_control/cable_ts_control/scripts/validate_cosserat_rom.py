"""Compare native FOM and MOR graphs before promoting a strain basis (selection on the
validation trajectories, one independent evaluation of the test trajectories)."""

from pathlib import Path
import shutil

import numpy as np
import yaml

from cable_identification import cosserat_model as cm
from cable_identification.cable_mor_training_scene import excitation_paths, prepare_episode, run_episode
from cable_identification.strain_basis import (
    ReductionSpec, file_sha256, project_strain, validate_reduction, weighted_norm,
)
from cable_ts_control.scripts.generate_sofa_dataset import parameter_vertices


def _angle_between(q_a, q_b):
    dots = np.abs(np.sum(q_a * q_b, axis=1)) / (np.linalg.norm(q_a, axis=1) * np.linalg.norm(q_b, axis=1))
    return 2. * np.arccos(np.clip(dots, 0., 1.))


def grasp_tracking(samples, cfg):
    """Each model's grasped frame against the commanded grasp pose (gripper (+) attachment
    offset, the target the coupling writes): position [m], orientation [rad], and the
    position the configured compliance predicts from the reaction, |F| / k."""
    frame, target = samples["grasp_frame_poses"], samples["grasp_targets"]
    force = np.linalg.norm(samples["grasp_wrenches"][:, :3], axis=1)
    return {"position_m": float(np.max(np.linalg.norm(frame[:, :3] - target[:, :3], axis=1))),
            "orientation_rad": float(np.max(_angle_between(frame[:, 3:], target[:, 3:]))),
            "compliance_m": float(np.max(force) / float(cfg["grasp_stiffness"]))}


def comparison_metrics(full, reduced, cfg, weights, full_time, reduced_time):
    """ROM against FOM on the same episode, plus each model's own grasp tracking.

    Shapes are compared at matching material coordinates (every centerline frame and the
    sparse markers); strains in the strain-energy norm AND per active component, so bending
    cannot hide an axial or shear error; the physical length is ``sum(l_i |v_i|)`` from the
    strains, the polyline chord of the frames is reported separately.
    """
    n_frames = reduced["frames_xy"].shape[1] // 2
    marker_error = reduced["shapes"] - full["shapes"]
    centerline_error = (reduced["frames_xy"] - full["frames_xy"]).reshape(len(full["frames_xy"]), n_frames, 2)
    tip_position = np.linalg.norm(reduced["tip_poses"][:, :3] - full["tip_poses"][:, :3], axis=1)
    strain_error = reduced["strain"] - full["strain"]
    components = {"bend": 2, "extension": 3, "shear": 4}
    n_components = len(cm.STRAIN_COMPONENTS)
    full_grasp, reduced_grasp = grasp_tracking(full, cfg), grasp_tracking(reduced, cfg)
    wrench_scale = max(np.max(np.linalg.norm(full["grasp_wrenches"], axis=1)), 1e-12)
    metrics = {
        "marker_rmse_m": float(np.sqrt(np.mean(marker_error ** 2))),
        "marker_max_m": float(np.max(np.linalg.norm(marker_error.reshape(len(marker_error), -1, 2), axis=2))),
        "centerline_rmse_m": float(np.sqrt(np.mean(np.sum(centerline_error ** 2, axis=2)))),
        "centerline_max_m": float(np.max(np.linalg.norm(centerline_error, axis=2))),
        "tip_position_m": float(np.max(tip_position)),
        "tip_orientation_rad": float(np.max(_angle_between(full["tip_poses"][:, 3:], reduced["tip_poses"][:, 3:]))),
        "strain_relative_error": float(weighted_norm(weights, strain_error)
                                       / max(weighted_norm(weights, full["strain"]), 1e-12)),
        "strain_rate_relative_error": float(
            weighted_norm(weights, reduced["strain_velocity"] - full["strain_velocity"])
            / max(weighted_norm(weights, full["strain_velocity"]), 1e-12)),
        "physical_length_error_m": float(np.max(np.abs(reduced["physical_length"] - full["physical_length"]))),
        "chord_length_error_m": float(np.max(np.abs(reduced["chain_length"] - full["chain_length"]))),
        "grasp_wrench_relative_error": float(
            np.max(np.linalg.norm(reduced["grasp_wrenches"] - full["grasp_wrenches"], axis=1)) / wrench_scale),
        "grasp_position_m": max(full_grasp["position_m"], reduced_grasp["position_m"]),
        "grasp_orientation_rad": max(full_grasp["orientation_rad"], reduced_grasp["orientation_rad"]),
        "fom_grasp": full_grasp, "rom_grasp": reduced_grasp,
        "fom_step_seconds": float(full_time), "rom_step_seconds": float(reduced_time),
        "speedup": float(full_time / reduced_time),
    }
    for name, component in components.items():
        rows = slice(component, None, n_components)
        reference = np.linalg.norm(full["strain"][:, rows])
        # a component the FOM barely uses (below 1e-6 rms strain) is compared in absolute terms
        scale = reference if reference > 1e-6 * np.sqrt(full["strain"][:, rows].size) else 1.0
        metrics[f"strain_{name}_relative_error"] = float(np.linalg.norm(strain_error[:, rows]) / scale)
    metrics["strain_component_relative_error"] = max(
        metrics[f"strain_{name}_relative_error"] for name in components)
    flat = [v for k, v in metrics.items() if k not in ("fom_grasp", "rom_grasp")]
    if not all(np.isfinite(value) for value in flat):
        raise ValueError("Non-finite FOM/ROM validation metric")
    return metrics


def passes(metrics, limits):
    if "diverged" in metrics:
        return False
    return all(metrics[name] <= limit for name, limit in limits.items() if name in metrics)


def projected_rollouts(cfg, parameters, path, control_dt, reduction, full, modes, weights, horizon):
    import Sofa.Core
    import Sofa.Simulation

    root = Sofa.Core.Node("projected_rollout_validation")
    one_step_errors, rollout_errors = [], []
    try:
        cable, controller = prepare_episode(root, cfg, parameters, path, control_dt, reduction=reduction)
        state = cable.save_state()
        for start in range(0, len(path) - horizon - 1, horizon):
            for duration, errors in ((1, one_step_errors), (horizon, rollout_errors)):
                state["modal"] = project_strain(modes, weights, full["strain"][start]).reshape(-1, 1)
                state["modal_velocity"] = project_strain(
                    modes, weights, full["strain_velocity"][start]).reshape(-1, 1)
                cable.restore_state(state)
                controller.step = start * controller.substeps
                for _ in range(duration * controller.substeps):
                    Sofa.Simulation.animate(root, cfg["timestep_s"])
                markers = np.asarray(cable.marker_positions())[:, :2].ravel()
                errors.append(markers - full["shapes"][start + duration])
    finally:
        Sofa.Simulation.unload(root)
    return {"one_step_marker_rmse_m": float(np.sqrt(np.mean(np.asarray(one_step_errors) ** 2))),
            "rollout_2s_marker_rmse_m": float(np.sqrt(np.mean(np.asarray(rollout_errors) ** 2)))}


def run_pair(cfg, values, path, control_dt, reduction, weights, reference, key):
    """FOM/ROM metrics of one episode; a diverging ROM fails the episode with NaN metrics
    (the FOM reference is cached per (vertex, trajectory))."""
    if key not in reference:
        reference[key] = run_episode(cfg, values, path, control_dt)
    full, full_time = reference[key]
    try:
        reduced, reduced_time = run_episode(cfg, values, path, control_dt, reduction=reduction)
        return comparison_metrics(full, reduced, cfg, weights, full_time, reduced_time)
    except ValueError as error:
        if "diverged" not in str(error):
            raise
        print(f"  {error}", flush=True)
        return {"diverged": str(error), "centerline_rmse_m": float("nan"), "grasp_position_m": float("nan"),
                "strain_extension_relative_error": float("nan"), "physical_length_error_m": float("nan"),
                "speedup": float("nan")}


def validate(directory, cfg, settings, bounds):
    """Select the smallest POD order that meets the mechanical tolerances on the validation
    trajectories (train trajectory 0 is run alongside as a sanity check), freeze it, then
    evaluate the independent test trajectories once. A failing test is reported as such."""
    from cable_ts_control.sofa_mor_pipeline import load_snapshot_manifest

    load_snapshot_manifest(directory, cfg, settings, bounds)
    with (directory / "candidates.yaml").open() as stream:
        candidates = yaml.safe_load(stream)
    paths = excitation_paths(cfg, settings)
    parameters = parameter_vertices(bounds) + [
        {name: float(np.mean(values)) for name, values in bounds.items()}]
    limits = settings["validation"]
    weights = cm.strain_weights(cfg)
    control_dt = settings["control_dt"]
    reference = {}
    selection_trajectories = [settings["train_trajectories"][0], *settings["validation_trajectories"]]
    test_trajectories = list(settings["test_trajectories"])
    if set(selection_trajectories) & set(test_trajectories):
        raise ValueError("Test trajectories must not take part in the order selection")
    report = {"passed": False, "limits": limits, "selection_trajectories": selection_trajectories,
              "test_trajectories": test_trajectories, "active_dimension": len(cm.active_components(cfg)),
              "candidates": []}
    report_path = directory / "cable_rom_validation.yaml"

    def episode_record(vertex_id, values, trajectory_id, metrics, passed):
        split = next(label for label in ("train", "validation", "test")
                     if trajectory_id in settings[f"{label}_trajectories"])
        return {"vertex_id": vertex_id, "parameters": values, "trajectory_id": trajectory_id,
                "split": split, "passed": bool(passed), **metrics}

    selected = None
    for candidate in sorted(candidates, key=lambda item: item["n_modes"]):
        reduction = ReductionSpec(str(directory / candidate["modes_path"]),
                                  str(directory / candidate["metadata_path"]), candidate["n_modes"])
        _, modes, metadata = validate_reduction(cfg, reduction)
        result = {"n_modes": reduction.n_modes, "strain_basis_sha256": metadata["strain_basis_sha256"],
                  "passed": True, "episodes": []}
        for vertex_id, values in enumerate(parameters):
            for trajectory_id in selection_trajectories:
                metrics = run_pair(cfg, values, paths[trajectory_id], control_dt, reduction, weights,
                                   reference, (vertex_id, trajectory_id))
                passed = passes(metrics, limits)
                result["passed"] &= passed
                result["episodes"].append(episode_record(vertex_id, values, trajectory_id, metrics, passed))
                print(f"ROM r={reduction.n_modes} vertex={vertex_id} trajectory={trajectory_id}: "
                      f"centerline={metrics['centerline_rmse_m'] * 1000:.3f} mm, "
                      f"grasp={metrics['grasp_position_m'] * 1000:.3f} mm, "
                      f"extension err={metrics['strain_extension_relative_error']:.3g}, "
                      f"length err={metrics['physical_length_error_m'] * 1000:.3f} mm, "
                      f"speedup={metrics['speedup']:.3f}, passed={passed}", flush=True)
        if result["passed"]:
            result["holds"] = []
            for vertex_id, values in enumerate(parameters):
                hold_path = np.repeat(paths[settings["validation_trajectories"][0]][-1:],
                                      round(limits["hold_seconds"] / control_dt), axis=0)
                metrics = run_pair(cfg, values, hold_path, control_dt, reduction, weights, reference,
                                   ("hold", vertex_id))
                passed = passes(metrics, limits)
                result["passed"] &= passed
                result["holds"].append({"vertex_id": vertex_id, "passed": bool(passed), **metrics})
            trajectory_id = settings["validation_trajectories"][0]
            full = reference[(len(parameters) - 1, trajectory_id)][0]
            short_metrics = projected_rollouts(
                cfg, parameters[-1], paths[trajectory_id], control_dt, reduction, full, modes,
                weights, round(limits["rollout_seconds"] / control_dt))
            result.update(short_metrics)
            result["passed"] &= all(value <= limits["marker_rmse_m"] for value in short_metrics.values())
        result["passed"] = bool(result["passed"])
        report["candidates"].append(result)
        with report_path.open("w") as stream:
            yaml.safe_dump(report, stream, sort_keys=False)
        if result["passed"]:
            selected = (reduction, metadata, result)
            break
    if selected is None:
        raise ValueError("No candidate passed FOM/ROM validation; FOM retained and downstream phases blocked")

    # The order is frozen here; the test trajectories are seen exactly once.
    reduction, metadata, result = selected
    test = {"n_modes": reduction.n_modes, "passed": True, "episodes": []}
    for vertex_id, values in enumerate(parameters):
        for trajectory_id in test_trajectories:
            metrics = run_pair(cfg, values, paths[trajectory_id], control_dt, reduction, weights,
                               reference, (vertex_id, trajectory_id))
            passed = passes(metrics, limits)
            test["passed"] &= passed
            test["episodes"].append(episode_record(vertex_id, values, trajectory_id, metrics, passed))
            print(f"TEST r={reduction.n_modes} vertex={vertex_id} trajectory={trajectory_id}: "
                  f"centerline={metrics['centerline_rmse_m'] * 1000:.3f} mm, "
                  f"grasp={metrics['grasp_position_m'] * 1000:.3f} mm, passed={passed}", flush=True)
    test["passed"] = bool(test["passed"])
    report.update(selected_order=reduction.n_modes, strain_basis_sha256=metadata["strain_basis_sha256"],
                  coordinate_change_only=reduction.n_modes >= report["active_dimension"], test=test,
                  passed=test["passed"])
    with report_path.open("w") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    if not test["passed"]:
        raise ValueError(f"order r={reduction.n_modes} selected on the validation trajectories fails the "
                         f"independent test trajectories {test_trajectories}; no other candidate is tried")
    shutil.copyfile(reduction.modes_path, directory / "cable_strain_modes.txt")
    shutil.copyfile(reduction.metadata_path, directory / "cable_strain_modes.yaml")
    return {"selected_order": reduction.n_modes, "active_dimension": report["active_dimension"],
            "coordinate_change_only": report["coordinate_change_only"],
            "validation_sha256": file_sha256(report_path)}


def main(argv=None):
    from cable_ts_control.sofa_mor_pipeline import main as pipeline_main
    return pipeline_main(["--phase", "mor_validate", *(argv or [])])
"""Compare native FOM and MOR graphs before promoting a strain basis."""

from pathlib import Path
import shutil

import numpy as np
import yaml

from cable_identification.cable_mor_training_scene import excitation_paths, prepare_episode, run_episode
from cable_identification.strain_basis import ReductionSpec, file_sha256, validate_reduction
from cable_ts_control.scripts.generate_sofa_dataset import parameter_vertices


def comparison_metrics(full, reduced, length, full_time, reduced_time):
    marker_error = reduced["shapes"] - full["shapes"]
    tip_position = np.linalg.norm(reduced["tip_poses"][:, :3] - full["tip_poses"][:, :3], axis=1)
    full_quat = full["tip_poses"][:, 3:]
    reduced_quat = reduced["tip_poses"][:, 3:]
    dots = np.abs(np.sum(full_quat * reduced_quat, axis=1)) / (
        np.linalg.norm(full_quat, axis=1) * np.linalg.norm(reduced_quat, axis=1))
    metrics = {
        "marker_rmse_m": float(np.sqrt(np.mean(marker_error ** 2))),
        "marker_max_m": float(np.max(np.linalg.norm(marker_error.reshape(-1, 7, 2), axis=2))),
        "tip_position_m": float(np.max(tip_position)),
        "tip_orientation_rad": float(np.max(2. * np.arccos(np.clip(dots, 0., 1.)))),
        "strain_relative_error": float(np.linalg.norm(reduced["strain"] - full["strain"])
                                       / max(np.linalg.norm(full["strain"]), 1e-12)),
        "chain_length_error_m": float(max(np.max(np.abs(full["chain_length"] - length)),
                                           np.max(np.abs(reduced["chain_length"] - length)))),
        "fom_step_seconds": float(full_time), "rom_step_seconds": float(reduced_time),
        "speedup": float(full_time / reduced_time),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite FOM/ROM validation metric")
    return metrics


def projected_rollouts(cfg, parameters, path, control_dt, reduction, full, modes, horizon):
    import Sofa.Core
    import Sofa.Simulation

    root = Sofa.Core.Node("projected_rollout_validation")
    one_step_errors, rollout_errors = [], []
    try:
        cable, controller = prepare_episode(root, cfg, parameters, path, control_dt, reduction=reduction)
        state = cable.save_state()
        for start in range(0, len(path) - horizon - 1, horizon):
            for duration, errors in ((1, one_step_errors), (horizon, rollout_errors)):
                state["modal"] = (modes.T @ full["strain"][start]).reshape(-1, 1)
                state["modal_velocity"] = (modes.T @ full["strain_velocity"][start]).reshape(-1, 1)
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


def validate(directory, cfg, settings, bounds):
    from cable_ts_control.sofa_mor_pipeline import load_snapshot_manifest

    load_snapshot_manifest(directory, cfg, settings, bounds)
    with (directory / "candidates.yaml").open() as stream:
        candidates = yaml.safe_load(stream)
    paths = excitation_paths(cfg, settings)
    parameters = parameter_vertices(bounds) + [
        {name: float(np.mean(values)) for name, values in bounds.items()}]
    limits = settings["validation"]
    reference = {}
    report = {"passed": False, "limits": limits, "candidates": []}
    report_path = directory / "cable_rom_validation.yaml"
    for candidate in sorted(candidates, key=lambda item: item["n_modes"]):
        reduction = ReductionSpec(str(directory / candidate["modes_path"]),
                                  str(directory / candidate["metadata_path"]), candidate["n_modes"])
        _, modes, metadata = validate_reduction(cfg, reduction)
        result = {"n_modes": reduction.n_modes, "strain_basis_sha256": metadata["strain_basis_sha256"],
                  "passed": True, "episodes": []}
        for vertex_id, values in enumerate(parameters):
            for trajectory_id in [settings["train_trajectories"][0],
                                  *settings["validation_trajectories"], *settings["test_trajectories"]]:
                key = (vertex_id, trajectory_id)
                if key not in reference:
                    reference[key] = run_episode(cfg, values, paths[trajectory_id], settings["control_dt"])
                full, full_time = reference[key]
                reduced, reduced_time = run_episode(
                    cfg, values, paths[trajectory_id], settings["control_dt"], reduction=reduction)
                metrics = comparison_metrics(full, reduced, cfg["length_m"], full_time, reduced_time)
                passed = all(metrics[name] <= limit for name, limit in limits.items() if name in metrics)
                result["passed"] &= passed
                split = next(label for label in ("train", "validation", "test")
                             if trajectory_id in settings[f"{label}_trajectories"])
                result["episodes"].append({"vertex_id": vertex_id, "parameters": values,
                                           "trajectory_id": trajectory_id, "split": split,
                                           "passed": bool(passed), **metrics})
                print(f"ROM r={reduction.n_modes} vertex={vertex_id} trajectory={trajectory_id}: "
                      f"markers={metrics['marker_rmse_m'] * 1000:.3f} mm, "
                      f"tip={metrics['tip_position_m'] * 1000:.3f} mm, "
                      f"speedup={metrics['speedup']:.3f}, passed={passed}", flush=True)
        if result["passed"]:
            result["holds"] = []
            for vertex_id, values in enumerate(parameters):
                hold_path = np.repeat(paths[settings["validation_trajectories"][0]][-1:],
                                      round(limits["hold_seconds"] / settings["control_dt"]), axis=0)
                full, full_time = run_episode(cfg, values, hold_path, settings["control_dt"])
                reduced, reduced_time = run_episode(cfg, values, hold_path, settings["control_dt"], reduction=reduction)
                metrics = comparison_metrics(full, reduced, cfg["length_m"], full_time, reduced_time)
                passed = all(metrics[name] <= limit for name, limit in limits.items() if name in metrics)
                result["passed"] &= passed
                result["holds"].append({"vertex_id": vertex_id, "passed": bool(passed), **metrics})
            trajectory_id = settings["validation_trajectories"][0]
            full = reference[(len(parameters) - 1, trajectory_id)][0]
            short_metrics = projected_rollouts(
                cfg, parameters[-1], paths[trajectory_id], settings["control_dt"], reduction, full, modes,
                round(limits["rollout_seconds"] / settings["control_dt"]))
            result.update(short_metrics)
            result["passed"] &= all(value <= limits["marker_rmse_m"] for value in short_metrics.values())
        result["passed"] = bool(result["passed"])
        report["candidates"].append(result)
        with report_path.open("w") as stream:
            yaml.safe_dump(report, stream, sort_keys=False)
        if result["passed"]:
            report.update(passed=True, selected_order=reduction.n_modes,
                          strain_basis_sha256=metadata["strain_basis_sha256"])
            shutil.copyfile(reduction.modes_path, directory / "cable_strain_modes.txt")
            shutil.copyfile(reduction.metadata_path, directory / "cable_strain_modes.yaml")
            with report_path.open("w") as stream:
                yaml.safe_dump(report, stream, sort_keys=False)
            return {"selected_order": reduction.n_modes, "validation_sha256": file_sha256(report_path)}
    raise ValueError("No candidate passed FOM/ROM validation; FOM retained and downstream phases blocked")


def main(argv=None):
    from cable_ts_control.sofa_mor_pipeline import main as pipeline_main
    return pipeline_main(["--phase", "mor_validate", *(argv or [])])
"""Gated SofaPython3 orchestration using one native Cosserat strain POD."""

import argparse
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

from cable_identification import cosserat_model as cm
from cable_identification.parameter_bounds import load_parameter_bounds
from cable_identification.strain_basis import (
    ReductionSpec, config_sha256, file_sha256, read_modes, read_state_file, validate_reduction,
)
from cable_ts_control.scripts.generate_sofa_dataset import parameter_vertices

# Only these settings define the recorded SOFA states; TS/LMI/PDC/observer
# settings may change without invalidating the snapshots.
SNAPSHOT_KEYS = ("control_dt", "seed", "n_trajectories", "control_steps", "hold_control_steps",
                 "train_trajectories", "validation_trajectories", "test_trajectories",
                 "radial_min", "radial_max", "angle_limit", "max_speed", "sweeps")


def snapshot_settings_sha256(settings):
    return config_sha256({key: settings[key] for key in SNAPSHOT_KEYS})


def snapshots(directory, cfg, settings, bounds):
    from cable_identification.cable_mor_training_scene import excitation_paths, run_episode

    splits = {label: settings[f"{label}_trajectories"] for label in ("train", "validation", "test")}
    identifiers = [identifier for group in splits.values() for identifier in group]
    if sorted(identifiers) != list(range(settings["n_trajectories"])):
        raise ValueError("Train/validation/test must be a disjoint whole-trajectory partition")
    if set(bounds) != {"EI", "rayleigh_stiffness"} or not cfg["planar"] or not cfg["grasp_tip"]:
        raise ValueError("This training configuration requires planar EI/Rayleigh vertices and a grasp")
    paths = excitation_paths(cfg, settings)
    vertices = parameter_vertices(bounds)
    state_directory = directory / "states"
    state_directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    episodes = []
    offset = 0
    for vertex_id, parameters in enumerate(vertices):
        for trajectory_id, path in enumerate(paths):
            state_file = state_directory / f"vertex_{vertex_id}_trajectory_{trajectory_id}.state"
            samples, mean_step = run_episode(cfg, parameters, path, settings["control_dt"], state_file)
            samples["vertex_ids"] = np.full(len(path), vertex_id)
            samples["trajectory_ids"] = np.full(len(path), trajectory_id)
            for key, values in samples.items():
                arrays.setdefault(key, []).append(values)
            split = next(label for label, group in splits.items() if trajectory_id in group)
            episodes.append({"vertex_id": vertex_id, "trajectory_id": trajectory_id,
                             "split": split, "state_file": str(state_file.relative_to(directory)),
                             "state_sha256": file_sha256(state_file),
                             "sample_start": offset, "sample_stop": offset + len(path),
                             "mean_step_seconds": mean_step})
            offset += len(path)
            print(f"WriteState vertex={vertex_id} trajectory={trajectory_id} split={split}: "
                  f"{len(path)} x {3 * cfg['number_of_sections']}, dt={settings['control_dt']}", flush=True)
    dataset = directory / "cable_mor_snapshots_fom.npz"
    np.savez_compressed(
        dataset, **{key: np.concatenate(values) for key, values in arrays.items()},
        parameter_names=np.asarray(sorted(bounds)),
        parameter_values=np.asarray([[vertex[name] for name in sorted(bounds)] for vertex in vertices]),
        marker_s_over_l=np.asarray(cfg["marker_s_over_l"]),
        cable_length_m=np.asarray([cfg["length_m"]]),
        timestep_s=np.asarray([settings["control_dt"]]),
        sofa_timestep_s=np.asarray([cfg["timestep_s"]]))
    training_state = directory / "training.state"
    with training_state.open("wb") as target:
        for episode in episodes:
            if episode["split"] == "train":
                with (directory / episode["state_file"]).open("rb") as source:
                    shutil.copyfileobj(source, target)
    manifest = {"schema_version": 1, "state_coordinates": "cosserat_strain",
                "config": cfg, "config_sha256": config_sha256(cfg),
                "settings": settings, "settings_sha256": snapshot_settings_sha256(settings),
                "parameter_bounds": bounds, "repo_commit": os.environ.get("CABLE_REPO_COMMIT", "unknown"),
                "dataset_sha256": file_sha256(dataset),
                "training_state_sha256": file_sha256(training_state), "episodes": episodes}
    with (directory / "snapshots.yaml").open("w") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)
    return {"samples": offset, "physical_vertices": len(vertices), "trajectories": len(paths)}


def load_snapshot_manifest(directory, cfg, settings, bounds):
    with (directory / "snapshots.yaml").open() as stream:
        manifest = yaml.safe_load(stream)
    if manifest["config_sha256"] != config_sha256(cfg) or manifest["settings_sha256"] != snapshot_settings_sha256(settings):
        raise ValueError("Snapshot configuration changed; regenerate the SOFA recordings")
    if config_sha256(manifest["parameter_bounds"]) != config_sha256(bounds):
        raise ValueError("Snapshot parameter envelope changed")
    for filename, key in (("cable_mor_snapshots_fom.npz", "dataset_sha256"),
                          ("training.state", "training_state_sha256")):
        if file_sha256(directory / filename) != manifest[key]:
            raise ValueError(f"Snapshot hash mismatch: {filename}")
    return manifest


def compute_modes(directory, cfg, settings, bounds):
    from cable_identification.mor_plugin_test import MOR_COMMIT
    from mor.reduction.script.ReadStateFilesAndComputeModes import readStateFilesAndComputeModes

    manifest = load_snapshot_manifest(directory, cfg, settings, bounds)
    positions, velocities = [], []
    reference = None
    for episode in manifest["episodes"]:
        if episode["split"] != "train":
            continue
        path = directory / episode["state_file"]
        if file_sha256(path) != episode["state_sha256"]:
            raise ValueError("Native training state hash mismatch")
        native = read_state_file(path, 3 * cfg["number_of_sections"])
        if reference is not None and not np.array_equal(reference, native["reference"]):
            raise ValueError("Training trajectories use different neutral references")
        reference = native["reference"]
        positions.append(native["strain"])
        velocities.append(native["strain_velocity"])
    strain = np.concatenate(positions) - reference
    strain_velocity = np.concatenate(velocities)
    if np.linalg.norm(strain) <= 1e-12:
        raise ValueError("No dynamic strain excitation in the training states")
    candidates_directory = directory / "candidates"
    candidates_directory.mkdir(exist_ok=True)
    candidates = []
    for tolerance in settings["pod_tolerances"]:
        if not 0. < tolerance < 1.:
            raise ValueError("POD residual tolerance must lie strictly between zero and one")
        modes_path = candidates_directory / f"tol_{tolerance:g}.txt"
        order = int(readStateFilesAndComputeModes(
            stateFilePath=str(directory / "training.state"), tol=float(tolerance),
            modesFileName=str(modes_path), addRigidBodyModes=None, verbose=True))
        if order < 1:
            raise ValueError("Official SOFA-MOR POD failed")
        modes = read_modes(modes_path)
        singular = np.loadtxt(directory / "Sdata.txt", ndmin=1)
        energies = np.cumsum(singular ** 2) / np.sum(singular ** 2)
        metadata = {
            "schema_version": 1, "state_coordinates": "cosserat_modal",
            "number_of_sections": cfg["number_of_sections"],
            "full_dimension": 3 * cfg["number_of_sections"], "n_modes": order,
            "component_order": ["kappa_x", "kappa_y", "kappa_z"],
            "active_components": list(range(2, 3 * cfg["number_of_sections"], 3)),
            "kappa_0": reference.tolist(), "singular_values": singular.tolist(),
            "cumulative_energy": energies.tolist(), "pod_tolerance": tolerance,
            "rank_relative_tolerance": settings["rank_relative_tolerance"],
            "orthogonality_tolerance": settings["basis_orthogonality_tolerance"],
            "projection_position_relative": float(np.linalg.norm(strain - strain @ modes @ modes.T)
                                                   / np.linalg.norm(strain)),
            "projection_velocity_relative": float(np.linalg.norm(
                strain_velocity - strain_velocity @ modes @ modes.T) / max(np.linalg.norm(strain_velocity), 1e-15)),
            "parameter_bounds": bounds, "train_trajectories": settings["train_trajectories"],
            "dataset_sha256": manifest["dataset_sha256"],
            "config_sha256": manifest["config_sha256"], "settings_sha256": manifest["settings_sha256"],
            "strain_basis_sha256": file_sha256(modes_path),
            "repo_commit": manifest["repo_commit"], "mor_commit": MOR_COMMIT,
            "velocity_alpha": settings["velocity_alpha"], "control_dt": settings["control_dt"],
            "marker_s_over_l": cfg["marker_s_over_l"],
        }
        metadata_path = modes_path.with_suffix(".yaml")
        with metadata_path.open("w") as stream:
            yaml.safe_dump(metadata, stream, sort_keys=False)
        validate_reduction(cfg, ReductionSpec(str(modes_path), str(metadata_path), order))
        if not any(candidate["n_modes"] == order for candidate in candidates):
            candidates.append({"n_modes": order, "modes_path": str(modes_path.relative_to(directory)),
                               "metadata_path": str(metadata_path.relative_to(directory)),
                               "position_projection_relative": metadata["projection_position_relative"],
                               "velocity_projection_relative": metadata["projection_velocity_relative"]})
        print(f"POD r={order}, position={metadata['projection_position_relative']:.5g}, "
              f"velocity={metadata['projection_velocity_relative']:.5g}", flush=True)
    with (directory / "candidates.yaml").open("w") as stream:
        yaml.safe_dump(candidates, stream, sort_keys=False)
    return candidates


def main(argv=None):
    from ament_index_python.packages import get_package_share_directory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=list(PHASES), required=True)
    parser.add_argument("--output-dir", default="/ros2_ws/artifacts/cable_mor")
    parser.add_argument("--config", default=str(Path(get_package_share_directory(
        "cable_identification")) / "config/cable_truth.yaml"))
    parser.add_argument("--settings", default=str(Path(get_package_share_directory(
        "cable_ts_control")) / "config/cable_mor.yaml"))
    parser.add_argument("--parameter-bounds", default=str(Path(get_package_share_directory(
        "cable_ts_control")) / "config/cable_parameter_bounds.yaml"))
    args = parser.parse_args(argv)
    label = f"CABLE_SOFA_{args.phase.upper()}"
    try:
        directory = Path(args.output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        cfg = cm.load_config(args.config)
        with open(args.settings) as stream:
            settings = yaml.safe_load(stream)
        bounds = load_parameter_bounds(args.parameter_bounds)
        result = PHASES[args.phase](directory, cfg, settings, bounds)
        print(f"{label}_PASSED: {result}")
        return 0
    except Exception as error:
        print(f"{label}_FAILED: {error}", file=sys.stderr)
        return 1


def promoted_reduction(directory, cfg):
    """The basis promoted by mor_validate; every downstream phase refuses to run without it."""
    metadata_path = directory / "cable_strain_modes.yaml"
    modes_path = directory / "cable_strain_modes.txt"
    if not metadata_path.is_file() or not modes_path.is_file():
        raise ValueError("no promoted strain basis: run cable_sofa_mor_validate first")
    with metadata_path.open() as stream:
        metadata = yaml.safe_load(stream)
    with (directory / "cable_rom_validation.yaml").open() as stream:
        report = yaml.safe_load(stream)
    if not report.get("passed") or report.get("strain_basis_sha256") != metadata["strain_basis_sha256"]:
        raise ValueError("cable_rom_validation.yaml does not certify the promoted basis")
    reduction = ReductionSpec(str(modes_path), str(metadata_path), int(metadata["n_modes"]))
    validate_reduction(cfg, reduction)
    return reduction, metadata


def modal_dataset(directory, cfg, settings, bounds):
    from cable_identification.cable_mor_training_scene import excitation_paths, run_episode
    from cable_ts_control.modal_contract import COORDINATES
    from cable_ts_control.state_filter import velocity_series

    reduction, metadata = promoted_reduction(directory, cfg)
    load_snapshot_manifest(directory, cfg, settings, bounds)
    paths = excitation_paths(cfg, settings)
    vertices = parameter_vertices(bounds)
    arrays = {}
    for vertex_id, parameters in enumerate(vertices):
        for trajectory_id, path in enumerate(paths):
            samples, _ = run_episode(cfg, parameters, path, settings["control_dt"], reduction=reduction)
            # The velocity the TS is trained on is the one the observer will produce online.
            samples["modal_velocities_state"] = velocity_series(
                samples["modal_positions"], settings["control_dt"], settings["velocity_alpha"])
            samples["vertex_ids"] = np.full(len(path), vertex_id)
            samples["trajectory_ids"] = np.full(len(path), trajectory_id)
            for key, values in samples.items():
                arrays.setdefault(key, []).append(values)
            print(f"ROM dataset vertex={vertex_id} trajectory={trajectory_id}: {len(path)} samples, "
                  f"r={reduction.n_modes}", flush=True)
    dataset = directory / "cable_dataset_rom.npz"
    np.savez_compressed(
        dataset, **{key: np.concatenate(values) for key, values in arrays.items()},
        parameter_names=np.asarray(sorted(bounds)),
        parameter_values=np.asarray([[vertex[name] for name in sorted(bounds)] for vertex in vertices]),
        marker_s_over_l=np.asarray(cfg["marker_s_over_l"]),
        cable_length_m=np.asarray([cfg["length_m"]]),
        timestep_s=np.asarray([settings["control_dt"]]),
        sofa_timestep_s=np.asarray([cfg["timestep_s"]]),
        velocity_alpha=np.asarray([settings["velocity_alpha"]]),
        state_coordinates=np.asarray(COORDINATES),
        strain_basis_sha256=np.asarray(metadata["strain_basis_sha256"]))
    return {"samples": int(sum(len(v) for v in arrays["vertex_ids"])), "n_modes": reduction.n_modes,
            "dataset_sha256": file_sha256(dataset)}


def modal_ts_identify(directory, cfg, settings, bounds):
    from cable_identification.modal_observation import ModalObservationModel
    from cable_ts_control.sofa_modal_ts_identification import CableModalTsIdentificationController

    reduction, metadata = promoted_reduction(directory, cfg)
    dataset_path = directory / "cable_dataset_rom.npz"
    dataset = np.load(dataset_path, allow_pickle=False)
    observation = ModalObservationModel(cfg, reduction)
    try:
        controller = observation.root.addObject(CableModalTsIdentificationController(
            observation, dataset, metadata, settings, file_sha256(dataset_path),
            name="modalTsIdentification"))
        payload, report = controller.identify()
    finally:
        observation.close()
    output = directory / "cable_ts_model_modal_rom.yaml"
    with output.open("w") as stream:
        yaml.safe_dump(payload, stream, default_flow_style=False, width=200)
    with (directory / "cable_ts_identification_report.yaml").open("w") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    if not report["passed"]:
        raise ValueError(f"held-out one-step {report['worst_one_step_marker_rmse_m'] * 1e3:.3f} mm / "
                         f"rollout {report['worst_rollout_marker_rmse_m'] * 1e3:.3f} mm exceed the "
                         f"limits {settings['ts']}")
    return {"n_modes": reduction.n_modes, "state_dim": payload["cable_ts_model"]["state_dim"],
            "worst_one_step_marker_rmse_m": report["worst_one_step_marker_rmse_m"],
            "worst_rollout_marker_rmse_m": report["worst_rollout_marker_rmse_m"]}


def modal_lmi(directory, cfg, settings, bounds):
    from cable_ts_control.modal_contract import COORDINATES, check_model_contract
    from cable_ts_control.scripts import solve_cable_ts_lmi
    from cable_ts_control.ts_model import CableTsModel

    _, metadata = promoted_reduction(directory, cfg)
    model_path = directory / "cable_ts_model_modal_rom.yaml"
    with model_path.open() as stream:
        section = yaml.safe_load(stream)["cable_ts_model"]
    check_model_contract(section, metadata, expected_coordinates=COORDINATES)
    output = directory / "cable_ts_gains_modal_rom.yaml"
    if output.exists():
        output.unlink()
    lmi = settings["lmi"]
    report_path = directory / "cable_lmi_report.yaml"
    status = solve_cable_ts_lmi.main(["--model", str(model_path), "--output", str(output),
                                      "--mode", lmi["mode"], "--eps", str(lmi["eps"]),
                                      "--gain-penalty", str(lmi["gain_penalty"]),
                                      "--backend", lmi["backend"], "--max-iters", str(lmi["max_iters"]),
                                      "--report", str(report_path)])
    if status != 0 or not output.exists():
        model = CableTsModel.from_dict({"cable_ts_model": section})
        worst = []
        for vertex, (a_group, b_group) in enumerate(solve_cable_ts_lmi.load_vertex_sets(model, section)):
            for rule, (a, b) in enumerate(zip(a_group, b_group)):
                ctrb = np.hstack([np.linalg.matrix_power(a, k) @ b for k in range(a.shape[0])])
                worst.append((float(np.max(np.abs(np.linalg.eigvals(a)))), vertex, rule,
                              int(np.linalg.matrix_rank(ctrb))))
        rho, vertex, rule, rank = max(worst)
        attempts = yaml.safe_load(report_path.open())["attempts"] if report_path.is_file() else []
        raise ValueError(f"dynamic LMI not certified ({lmi['mode']} conditions, {lmi['backend']} backend, "
                         f"{lmi['max_iters']} iterations); no gains written. Attempts: {attempts}. "
                         f"Worst open-loop rho(A)={rho:.4f} at vertex {vertex} rule {rule}, "
                         f"controllability rank {rank}/{model.state_dim}")
    with output.open() as stream:
        gains = yaml.safe_load(stream)
    gains["cable_ts_model"].update({
        "state_coordinates": COORDINATES, "strain_basis_sha256": section["strain_basis_sha256"],
        "n_modes": int(section["n_modes"]), "ts_model_sha256": file_sha256(model_path)})
    with output.open("w") as stream:
        yaml.safe_dump(gains, stream, default_flow_style=False, width=200)
    verification = gains["cable_ts_model"]["verification"]
    return {"certificate": gains["cable_ts_model"]["certificate"],
            "worst_spectral_radius": verification["worst_spectral_radius"],
            "worst_diagonal_residual": verification["worst_diagonal_residual"]}


def load_certified_model(directory, cfg):
    from cable_ts_control.modal_contract import COORDINATES, check_model_contract
    from cable_ts_control.ts_model import CableTsModel

    _, metadata = promoted_reduction(directory, cfg)
    model_path, gains_path = directory / "cable_ts_model_modal_rom.yaml", directory / "cable_ts_gains_modal_rom.yaml"
    if not gains_path.is_file():
        raise ValueError("no certified PDC gains (cable_sofa_modal_lmi did not pass)")
    with model_path.open() as stream:
        section = yaml.safe_load(stream)["cable_ts_model"]
    with gains_path.open() as stream:
        gains = yaml.safe_load(stream)["cable_ts_model"]
    section["sha256"] = file_sha256(model_path)
    check_model_contract(section, metadata, gains, expected_coordinates=COORDINATES)
    return CableTsModel.load(str(model_path), str(gains_path)), section, metadata


def settle_equilibrium(cfg, parameters, gripper, control_dt, reduction, seconds):
    """Hold the gripper at ``gripper``; return ``(a_eq, markers)`` of the settled ROM."""
    from cable_identification.cable_mor_training_scene import run_episode

    path = np.repeat(np.asarray(gripper, dtype=float)[None], round(seconds / control_dt), axis=0)
    samples, _ = run_episode(cfg, parameters, path, control_dt, reduction=reduction)
    if np.linalg.norm(samples["modal_velocities_sofa"][-1]) > 1e-4:
        raise ValueError("cable did not settle at the requested gripper pose")
    return samples["modal_positions"][-1], samples["shapes"][-1]


def closed_loop(cfg, parameters, reduction, model, section, target, start, settings, state_source_factory=None):
    import Sofa.Core
    import Sofa.Simulation
    from cable_identification.cable_mor_training_scene import prepare_episode
    from cable_ts_control.sofa_pdc_controller import CablePdcSofaController

    pdc = settings["pdc"]
    root = Sofa.Core.Node("cable_pdc_closed_loop")
    try:
        hold = np.repeat(np.asarray(start, dtype=float)[None], 2, axis=0)
        cable, coupling = prepare_episode(root, cfg, parameters, hold, settings["control_dt"],
                                          reduction=reduction, controller=False)
        for _ in range(round(pdc["settle_seconds"] / cfg["timestep_s"])):
            coupling.update_grasp([*start, 0., 0., 0., 0., 1.], 1.0)
            Sofa.Simulation.animate(root, cfg["timestep_s"])
        state_source = state_source_factory(cable) if state_source_factory else None
        controller = root.addObject(CablePdcSofaController(
            cable, coupling, model, target, settings["control_dt"], section["velocity_alpha"],
            section["gripper_reference"], start, pdc["max_linear_vel"], state_source, name="cablePdc"))
        for _ in range(round(pdc["duration_seconds"] / cfg["timestep_s"])):
            Sofa.Simulation.animate(root, cfg["timestep_s"])
        log = controller.arrays()
        log["final_markers"] = np.asarray(cable.marker_positions())[:, :2].ravel()
        log["final_modal"] = cable.modal_mo.position.value.ravel().copy() if cable.modal_mo is not None \
            else None
        frames = np.asarray(cable.frame_poses())
        log["chain_length"] = float(np.linalg.norm(np.diff(frames[:, :3], axis=0), axis=1).sum())
    finally:
        Sofa.Simulation.unload(root)
    return log


def evaluate_closed_loop(log, target_markers, cfg, settings, dt):
    from cable_ts_control.scripts.identify_ts_vertices import check_input_alignment

    pdc = settings["pdc"]
    if not all(np.all(np.isfinite(v)) for k, v in log.items() if k != "final_modal"):
        raise ValueError("non-finite value in the closed-loop log")
    # u[k] must be what moved the gripper between k and k+1 (dataset convention).
    check_input_alignment({"gripper": log["gripper"], "commands": log["command"],
                           "vertex_ids": np.zeros(len(log["gripper"])),
                           "trajectory_ids": np.zeros(len(log["gripper"]))}, dt)
    radius = np.linalg.norm(log["gripper"], axis=1) / cfg["length_m"]
    lyapunov = log["lyapunov"]
    metrics = {
        "initial_lyapunov": float(lyapunov[0]), "final_lyapunov": float(lyapunov[-1]),
        "lyapunov_decrease_fraction": float(np.mean(np.diff(lyapunov) <= 0.0)),
        "initial_error_norm": float(log["error_norm"][0]), "final_error_norm": float(log["error_norm"][-1]),
        "final_marker_rmse_m": float(np.sqrt(np.mean((log["final_markers"] - target_markers) ** 2))),
        "max_command_m_s": float(np.max(np.linalg.norm(log["command"], axis=1))),
        "gripper_radius_min": float(radius.min()), "gripper_radius_max": float(radius.max()),
        "chain_length_error_m": float(abs(log["chain_length"] - cfg["length_m"])),
    }
    metrics["passed"] = bool(
        metrics["final_marker_rmse_m"] <= pdc["final_marker_rmse_m"]
        and metrics["final_lyapunov"] <= pdc["lyapunov_ratio"] * metrics["initial_lyapunov"]
        and metrics["max_command_m_s"] <= pdc["max_linear_vel"] + 1e-12
        and 0.92 <= metrics["gripper_radius_min"] and metrics["gripper_radius_max"] <= 0.99 + 1e-9
        and metrics["chain_length_error_m"] <= settings["validation"]["chain_length_error_m"])
    return metrics


def pdc_test(directory, cfg, settings, bounds):
    from cable_identification.modal_observation import ModalObservationModel
    from cable_ts_control.modal_contract import target_state
    from cable_ts_control.modal_observer import ModalObserver

    model, section, metadata = load_certified_model(directory, cfg)
    reduction, _ = promoted_reduction(directory, cfg)
    pdc = settings["pdc"]
    parameters = {name: float(np.mean(values)) for name, values in bounds.items()}
    length = cfg["length_m"]
    start = length * pdc["start_radius"] * np.array([np.cos(pdc["start_bearing"]), np.sin(pdc["start_bearing"])])
    goal = length * pdc["goal_radius"] * np.array([np.cos(pdc["goal_bearing"]), np.sin(pdc["goal_bearing"])])
    a_goal, goal_markers = settle_equilibrium(cfg, parameters, goal, settings["control_dt"], reduction,
                                             pdc["settle_seconds"])
    target = target_state(np.concatenate([a_goal, goal - np.asarray(section["gripper_reference"])]),
                          model.n_modes, model.state_dim)
    results = {"target_gripper": goal.tolist(), "start_gripper": start.tolist()}
    log = closed_loop(cfg, parameters, reduction, model, section, target, start, settings)
    results["rom_direct_state"] = evaluate_closed_loop(log, goal_markers, cfg, settings, settings["control_dt"])
    print(f"PDC on ROM (direct modal state): {results['rom_direct_state']}", flush=True)

    # Transferable check: FOM truth, state estimated from noisy, partially occluded markers.
    observation = ModalObservationModel(cfg, reduction)
    rng = np.random.default_rng(settings["observer"]["seed"])
    observer_settings = settings["observer"]

    def estimated_state(cable):
        observer = ModalObserver(observation.markers, model.n_modes, observer_settings["marker_std_m"],
                                 observer_settings["prior_std"], observer_settings["max_iterations"])
        memory = {"a": np.zeros(model.n_modes)}

        def source(controller):
            markers = np.asarray(cable.marker_positions())[:, :2]
            noisy = markers + rng.normal(0.0, observer_settings["marker_std_m"], markers.shape)
            valid = rng.random(len(markers)) >= observer_settings["dropout"]
            valid[rng.integers(len(markers))] = True
            prior = memory["a"]
            if controller is not None and controller.log["state"]:
                # Prior = TS prediction from the last state and the command that drove it.
                prior = model.predict(controller.log["state"][-1], controller.command)[:model.n_modes]
            memory["a"], _ = observer.update(noisy.ravel(), valid, prior)
            return memory["a"]
        for _ in range(observer_settings["warm_start_updates"]):
            source(None)
        return source
    try:
        log = closed_loop(cfg, parameters, None, model, section, target, start, settings, estimated_state)
    finally:
        observation.close()
    results["fom_observer_state"] = evaluate_closed_loop(log, goal_markers, cfg, settings, settings["control_dt"])
    print(f"PDC on FOM (observer state): {results['fom_observer_state']}", flush=True)
    results["passed"] = bool(results["rom_direct_state"]["passed"] and results["fom_observer_state"]["passed"])
    with (directory / "cable_pdc_validation.yaml").open("w") as stream:
        yaml.safe_dump(results, stream, sort_keys=False)
    if results["passed"]:
        with (directory / "cable_target_modal_rom.yaml").open("w") as stream:
            yaml.safe_dump({"cable_target": {
                "frame": "cable_fixture_frame", "state_coordinates": "cosserat_modal",
                "strain_basis_sha256": metadata["strain_basis_sha256"],
                "modal_coordinates": target[:model.n_modes].tolist() + target[2 * model.n_modes:].tolist(),
                "shape_tolerance_m": 0.01, "hold_time_s": 2.0}}, stream, sort_keys=False)
    if not results["passed"]:
        raise ValueError(f"closed loop outside the limits {pdc}: {results}")
    return {key: results[key]["final_marker_rmse_m"] for key in ("rom_direct_state", "fom_observer_state")}


def modal_observer_test(directory, cfg, settings, bounds):
    import time
    from cable_identification.cable_mor_training_scene import excitation_paths, run_episode
    from cable_identification.modal_observation import ModalObservationModel
    from cable_ts_control.modal_contract import build_modal_state
    from cable_ts_control.modal_observer import ModalObserver
    from cable_ts_control.state_filter import ModalVelocityFilter, velocity_series
    from cable_ts_control.ts_model import CableTsModel

    reduction, metadata = promoted_reduction(directory, cfg)
    obs = settings["observer"]
    model_path = directory / "cable_ts_model_modal_rom.yaml"
    model = section = None
    if model_path.is_file():
        model = CableTsModel.load(str(model_path))
        with model_path.open() as stream:
            section = yaml.safe_load(stream)["cable_ts_model"]
    parameters = {name: float(np.mean(values)) for name, values in bounds.items()}
    path = excitation_paths(cfg, settings)[settings["validation_trajectories"][0]]
    # FOM is the truth; a = Phi^T kappa is the diagnostic oracle.
    truth, _ = run_episode(cfg, parameters, path, settings["control_dt"])
    observation = ModalObservationModel(cfg, reduction)
    try:
        a_true = np.stack([observation.project(strain) for strain in truth["strain"]])
        a_dot_true = velocity_series(a_true, settings["control_dt"], settings["velocity_alpha"])
        rng = np.random.default_rng(obs["seed"])
        report = {}
        for scenario, dropout in (("full_visibility", 0.0), ("occlusions", obs["dropout"])):
            observer = ModalObserver(observation.markers, reduction.n_modes, obs["marker_std_m"],
                                     obs["prior_std"], obs["max_iterations"])
            estimates, ranks, conditions, durations = [], [], [], []
            a_hat = np.zeros(reduction.n_modes)
            filt = ModalVelocityFilter(reduction.n_modes, settings["velocity_alpha"])
            gripper_reference = np.asarray(section["gripper_reference"]) if section else truth["gripper"].mean(axis=0)
            for k, markers in enumerate(truth["shapes"]):
                noisy = markers + rng.normal(0.0, obs["marker_std_m"], markers.shape)
                valid = rng.random(observation.n_markers) >= dropout
                valid[rng.integers(observation.n_markers)] = True
                prior = a_hat
                if model is not None and k > 0:
                    x = build_modal_state(a_hat, filt.velocity, truth["gripper"][k - 1], gripper_reference)
                    prior = model.predict(x, truth["commands"][k - 1])[:reduction.n_modes]
                started = time.perf_counter()
                a_hat, info = observer.update(noisy, valid, prior)
                durations.append(time.perf_counter() - started)
                filt.update(a_hat, settings["control_dt"])
                estimates.append(a_hat)
                ranks.append(info["rank"])
                conditions.append(info["condition"])
            estimates = np.asarray(estimates)
            a_dot_hat = velocity_series(estimates, settings["control_dt"], settings["velocity_alpha"])
            reconstructed = np.stack([observation.markers(a) for a in estimates])
            span = np.ptp(a_true, axis=0).max()
            report[scenario] = {
                "rank_min": int(min(ranks)), "rank_max": int(max(ranks)),
                "condition_median": float(np.median(conditions)), "condition_max": float(np.max(conditions)),
                "estimate_marker_rmse_m": float(np.sqrt(np.mean((reconstructed - truth["shapes"]) ** 2))),
                "modal_rmse": float(np.sqrt(np.mean((estimates - a_true) ** 2))),
                "modal_rmse_relative_to_range": float(np.sqrt(np.mean((estimates - a_true) ** 2)) / span),
                "modal_velocity_rmse": float(np.sqrt(np.mean((a_dot_hat - a_dot_true) ** 2))),
                "update_seconds_mean": float(np.mean(durations)), "update_seconds_max": float(np.max(durations)),
                "prior": "ts_model" if model is not None else "previous_estimate",
            }
            print(f"observer {scenario}: {report[scenario]}", flush=True)
    finally:
        observation.close()
    report["n_modes"] = reduction.n_modes
    report["n_measurements"] = 2 * observation.n_markers
    report["passed"] = bool(all(
        entry["estimate_marker_rmse_m"] <= obs["max_estimate_marker_rmse_m"]
        and entry["update_seconds_mean"] <= settings["control_dt"]
        for entry in (report["full_visibility"], report["occlusions"])))
    with (directory / "cable_modal_observer.yaml").open("w") as stream:
        yaml.safe_dump({"strain_basis_sha256": metadata["strain_basis_sha256"], "settings": obs,
                        "report": report}, stream, sort_keys=False)
    if not report["passed"]:
        raise ValueError(f"observer outside the limits: {report}")
    return {key: report[key]["estimate_marker_rmse_m"] for key in ("full_visibility", "occlusions")}


def mor_pipeline(directory, cfg, settings, bounds):
    from cable_identification.mor_plugin_test import check_mapping

    check_mapping()
    summary = {"mor_plugin_test": "PASSED"}
    for name in ("mor_snapshots", "mor_compute_modes", "mor_validate", "modal_dataset",
                 "modal_ts_identify", "modal_observer_test", "modal_lmi", "pdc_test"):
        try:
            result = PHASES[name](directory, cfg, settings, bounds)
        except Exception as error:
            summary[name] = f"FAILED: {error}"
            raise ValueError(f"stopped at {name}: {error}; gates so far {summary}") from error
        summary[name] = "PASSED"
        print(f"CABLE_SOFA_{name.upper()}_PASSED: {result}", flush=True)
    return summary


PHASES = {"mor_snapshots": snapshots, "mor_compute_modes": compute_modes,
          "modal_dataset": modal_dataset, "modal_ts_identify": modal_ts_identify,
          "modal_lmi": modal_lmi, "pdc_test": pdc_test, "modal_observer_test": modal_observer_test,
          "mor_pipeline": mor_pipeline}


def _register_validate():
    from cable_ts_control.scripts.validate_cosserat_rom import validate
    PHASES["mor_validate"] = validate


_register_validate()


if __name__ == "__main__":
    sys.exit(main())
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
                "settings": settings, "settings_sha256": config_sha256(settings),
                "parameter_bounds": bounds, "repo_commit": os.environ.get("CABLE_REPO_COMMIT", "unknown"),
                "dataset_sha256": file_sha256(dataset),
                "training_state_sha256": file_sha256(training_state), "episodes": episodes}
    with (directory / "snapshots.yaml").open("w") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)
    return {"samples": offset, "physical_vertices": len(vertices), "trajectories": len(paths)}


def load_snapshot_manifest(directory, cfg, settings, bounds):
    with (directory / "snapshots.yaml").open() as stream:
        manifest = yaml.safe_load(stream)
    if manifest["config_sha256"] != config_sha256(cfg) or manifest["settings_sha256"] != config_sha256(settings):
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
    parser.add_argument("--phase", choices=["mor_snapshots", "mor_compute_modes", "mor_validate"], required=True)
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
        from cable_ts_control.scripts.validate_cosserat_rom import validate
        phases = {"mor_snapshots": snapshots, "mor_compute_modes": compute_modes, "mor_validate": validate}
        result = phases[args.phase](directory, cfg, settings, bounds)
        print(f"{label}_PASSED: {result}")
        return 0
    except Exception as error:
        print(f"{label}_FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
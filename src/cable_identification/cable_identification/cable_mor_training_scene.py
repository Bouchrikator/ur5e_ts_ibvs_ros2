"""SOFA-owned excitation and native WriteState recording for the strain POD."""

import math
import os
from pathlib import Path
import time

import numpy as np
import Sofa.Core
import Sofa.Simulation

from cable_identification import cosserat_model as cm
from cable_identification.coupling import GraspCoupling
from cable_identification.strain_basis import read_state_file


def excitation_paths(cfg, settings):
    from cable_ts_control.scripts.generate_sofa_dataset import excitation_path

    rng = np.random.default_rng(settings["seed"])
    paths = []
    for _ in range(settings["n_trajectories"]):
        path = excitation_path(
            rng, settings["control_steps"], settings["control_dt"], cfg["length_m"],
            settings["radial_min"], settings["radial_max"], settings["angle_limit"],
            settings["max_speed"], settings["sweeps"])
        radius = np.clip(np.linalg.norm(path, axis=1),
                         settings["radial_min"] * cfg["length_m"],
                         settings["radial_max"] * cfg["length_m"])
        angle = np.arctan2(path[:, 1], path[:, 0])
        for index in range(1, len(path)):
            radial_change = radius[index] - radius[index - 1]
            angle_change = angle[index] - angle[index - 1]
            arc_bound = abs(radial_change) + max(radius[index], radius[index - 1]) * abs(angle_change)
            fraction = min(1., settings["max_speed"] * settings["control_dt"] / max(arc_bound, 1e-15))
            radius[index] = radius[index - 1] + fraction * radial_change
            angle[index] = angle[index - 1] + fraction * angle_change
        path = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
        hold = settings["hold_control_steps"]
        if hold < 1 or 2 * hold >= len(path):
            raise ValueError("Holds must leave a nonempty dynamic excitation interval")
        path[:hold] = path[hold]
        path[-hold:] = path[-hold]
        radius = np.linalg.norm(path, axis=1) / cfg["length_m"]
        if np.any(radius < 0.92 - 1e-12) or np.any(radius > 0.99 + 1e-12):
            raise ValueError("Excitation leaves the safe annulus or reaches the grasp clamp")
        if np.max(np.linalg.norm(np.diff(path, axis=0), axis=1)) > (
                settings["max_speed"] * settings["control_dt"] + 1e-12):
            raise ValueError("Excitation exceeds its velocity bound")
        paths.append(path)
    return paths


class CableMORSnapshotController(Sofa.Core.Controller):
    def __init__(self, cable, coupling, path, control_dt, **kwargs):
        super().__init__(**kwargs)
        self.cable = cable
        self.coupling = coupling
        self.path = np.asarray(path, dtype=float)
        self.control_dt = float(control_dt)
        self.sofa_dt = float(cable.cfg["timestep_s"])
        self.substeps = round(self.control_dt / self.sofa_dt)
        if self.substeps < 1 or abs(self.substeps * self.sofa_dt - self.control_dt) > 1e-12:
            raise ValueError("Control period must be an exact multiple of the SOFA timestep")
        self.step = 0
        self.samples = {key: [] for key in (
            "shapes", "commands", "gripper", "strain", "strain_velocity",
            "tip_poses", "chain_length")}
        if getattr(cable, "modal_mo", None) is not None:
            self.samples.update(modal_positions=[], modal_velocities_sofa=[])

    def onAnimateBeginEvent(self, event):
        sample = min(self.step // self.substeps, len(self.path) - 1)
        point = self.path[sample]
        following = self.path[min(sample + 1, len(self.path) - 1)]
        if self.step % self.substeps == 0 and self.step // self.substeps < len(self.path):
            frames = np.asarray(self.cable.frame_poses())
            values = {
                "shapes": np.asarray(self.cable.marker_positions())[:, :2].ravel(),
                "commands": (following - point) / self.control_dt,
                "gripper": point.copy(),
                "strain": self.cable.strain_mo.position.value.copy().ravel(),
                "strain_velocity": self.cable.strain_mo.velocity.value.copy().ravel(),
                "tip_poses": frames[-1].copy(),
                "chain_length": np.linalg.norm(np.diff(frames[:, :3], axis=0), axis=1).sum(),
            }
            if getattr(self.cable, "modal_mo", None) is not None:
                values["modal_positions"] = self.cable.modal_mo.position.value.copy().ravel()
                values["modal_velocities_sofa"] = self.cable.modal_mo.velocity.value.copy().ravel()
            if any(not np.all(np.isfinite(value)) for value in values.values()):
                raise ValueError("Non-finite SOFA state during excitation")
            for key, value in values.items():
                self.samples[key].append(value)
        fraction = (self.step % self.substeps + 1) / self.substeps
        target = point + fraction * (following - point)
        self.coupling.update_grasp([*target, 0., 0., 0., 0., 1.],
                                  1. + (self.step + 1) * self.sofa_dt)
        self.step += 1


def prepare_episode(root, cfg, parameters, path, control_dt, state_file=None, reduction=None,
                    controller=True):
    extra = ["Sofa.Component.Playback"] if state_file else []
    if reduction is not None:
        extra.append("ModelOrderReduction")
    cm.prepare_root(root, cfg, extra_plugins=extra)
    cable = (cm.build_cable(root, cfg) if reduction is None
             else cm.build_cable(root, cfg, reduction=reduction))
    coupling = GraspCoupling(cable, attach_mode="explicit")
    Sofa.Simulation.init(root)
    coupling.on_fixture([0., 0., 0., 0., 0., 0., 1.])
    for name, value in parameters.items():
        cable.set_parameter(name, value)
    coupling.request_attach()
    coupling.update_grasp(cable.tip_pose(), 0.)
    coupling.ramp_s = 0.
    if not coupling.latched:
        raise ValueError("The SOFA training cable did not attach")
    start = np.asarray(cable.tip_pose()[:2])
    warm_steps = max(1, math.ceil(np.linalg.norm(path[0] - start) / (0.1 * cfg["timestep_s"])))
    for index in range(warm_steps + 200):
        fraction = min(1., (index + 1) / warm_steps)
        point = start + fraction * (path[0] - start)
        coupling.update_grasp([*point, 0., 0., 0., 0., 1.], (index + 1) * cfg["timestep_s"])
        Sofa.Simulation.animate(root, cfg["timestep_s"])
    if state_file:
        writer = cable.strain_mo.getContext().addObject(
            "WriteState", name="strainStateWriter", filename=str(Path(state_file).resolve()),
            time=[float(root.time.value)], period=control_dt,
            writeX=True, writeX0=True, writeV=True, writeF=False)
        writer.init()
    if not controller:
        return cable, coupling
    controller = root.addObject(CableMORSnapshotController(
        cable, coupling, path, control_dt, name="cableModalExcitation"))
    return cable, controller


# The ROM dataset phase drives the same controller: it records a and a_dot when the
# cable carries a modal state, so no second recorder exists.
CableModalDatasetController = CableMORSnapshotController


def run_episode(cfg, parameters, path, control_dt, state_file=None, reduction=None):
    root = Sofa.Core.Node("cable_mor_episode")
    try:
        cable, controller = prepare_episode(
            root, cfg, parameters, path, control_dt, state_file, reduction)
        start_time = float(root.time.value)
        started = time.perf_counter()
        for _ in range(len(path) * controller.substeps):
            Sofa.Simulation.animate(root, cfg["timestep_s"])
        mean_step = (time.perf_counter() - started) / (len(path) * controller.substeps)
        samples = {key: np.asarray(value) for key, value in controller.samples.items()}
    finally:
        Sofa.Simulation.unload(root)
    if state_file:
        native = read_state_file(state_file, 3 * cfg["number_of_sections"])
        if native["times"].shape != (len(path),):
            raise ValueError(f"WriteState wrote {len(native['times'])}, expected {len(path)} samples")
        np.testing.assert_allclose(native["times"], start_time + np.arange(len(path)) * control_dt,
                       rtol=0., atol=1e-8)
        np.testing.assert_allclose(native["strain"], samples["strain"], rtol=1e-5, atol=1e-6)
        if np.any(native["reference"] != 0.):
            raise ValueError("The current cable requires the physical zero-strain reference")
        inactive = np.arange(3 * cfg["number_of_sections"]) % 3 != 2
        if cfg["planar"] and np.max(np.abs(native["strain"][:, inactive])) > 1e-10:
            raise ValueError("FOM planar strain constraint was not enforced")
        samples.update(strain=native["strain"], strain_velocity=native["strain_velocity"])
    return samples, mean_step


def createScene(rootNode):
    from ament_index_python.packages import get_package_share_directory
    import yaml

    control_share = Path(get_package_share_directory("cable_ts_control"))
    plant_share = Path(get_package_share_directory("cable_identification"))
    cfg = cm.load_config(os.environ.get("CABLE_CONFIG", str(plant_share / "config/cable_truth.yaml")))
    with open(control_share / "config/cable_mor.yaml") as stream:
        settings = yaml.safe_load(stream)
    path = excitation_paths(cfg, settings)[0]
    prepare_episode(rootNode, cfg, {}, path, settings["control_dt"],
                    os.environ.get("CABLE_MOR_STATE_FILE", "/tmp/cable_mor_training.state"))
    return rootNode
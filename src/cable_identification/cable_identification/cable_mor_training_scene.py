"""SOFA-owned excitation and native WriteState recording for the strain POD.

Excitation = gripper position on the declared radial/angular range (bending, tensile
extension beyond the rest length, reversals) coupled with a gripper yaw sweep: the
yaw is the chord bearing plus a relative yaw inside ``yaw_limit``. Pulling the cable
taut at a large bearing (or with the gripper far from tangent) forces a bending
boundary layer of length sqrt(EI/T) ~ 15-30 mm at the clamps (kappa > 10 /m, outside
the linear law's range and below the section size), so the bearing and the relative
yaw taper from their slack limits to ``angle_limit_taut`` between ``radial_bend`` and
the rest length. Every path is rate-limited (``max_speed``, ``max_yaw_rate``) and held
at both ends. Paths have columns ``(x, y, yaw)``; ``gripper_pose`` accepts the older
two-column form (yaw 0) used by the control stages.
"""

import math
import os
from pathlib import Path
import time

import numpy as np
import Sofa.Core
import Sofa.Simulation

from cable_identification import cosserat_model as cm
from cable_identification.coupling import GraspCoupling
from cable_identification.dynamics_dump import pose_delta
from cable_identification.strain_basis import read_state_file


def gripper_pose(point):
    """Excitation sample ``(x, y[, yaw])`` -> planar Rigid3d pose ``[x y 0 qx qy qz qw]``."""
    yaw = float(point[2]) if len(point) > 2 else 0.0
    return [float(point[0]), float(point[1]), 0.0, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


def physical_length(strain, section_lengths):
    """Current rod length ``sum(l_i |v_i|)`` from the total translational strain
    ``v_i = (1 + eps_x, eps_y, eps_z)`` of every section (rows of ``strain``: states)."""
    strain = np.asarray(strain, dtype=float).reshape(-1, len(section_lengths), len(cm.STRAIN_COMPONENTS))
    translational = strain[:, :, 3:].copy()
    translational[:, :, 0] += 1.0
    return np.linalg.norm(translational, axis=2) @ np.asarray(section_lengths, dtype=float)


def sweep_path(rng, n_steps, low, high, cycles, rate_limit=None, dither_scale=0.35):
    """Deterministic sweep over [low, high] with dither and reversals: both boundaries and the
    interior are visited (the angle generator's design; the legacy radius was an AR(1)
    filter whose spread, +-0.003 L, never reached the declared range)."""
    step = np.arange(n_steps)
    mid, half = 0.5 * (low + high), 0.5 * (high - low)
    base = mid + half * np.sin(2.0 * np.pi * cycles * step / n_steps + rng.uniform(0.0, 2.0 * np.pi))
    dither, current = np.zeros(n_steps), 0.0
    for k in range(n_steps):
        current = 0.85 * current + 0.15 * rng.normal(0.0, dither_scale * half)
        dither[k] = current
    values = np.clip(base + dither, low, high)
    if rate_limit is not None:
        for k in range(1, n_steps):
            values[k] = values[k - 1] + np.clip(values[k] - values[k - 1], -rate_limit, rate_limit)
    return values


def excitation_paths(cfg, settings):
    from cable_ts_control.scripts.generate_sofa_dataset import excitation_path

    rng = np.random.default_rng(settings["seed"])
    paths = []
    for _ in range(settings["n_trajectories"]):
        path = excitation_path(
            rng, settings["control_steps"], settings["control_dt"], cfg["length_m"],
            settings["radial_min"], settings["radial_max"], settings["angle_limit"],
            settings["max_speed"], settings["sweeps"])
        # the radius sweeps the declared range (bent at radial_min, taut and stretched beyond
        # 1.0 at radial_max); the yaw sweep is incommensurate with both so the three are coupled
        # but not phase-locked. Bearing and relative yaw taper towards the taut limit.
        # small radial dither: the sweep must really reach the taut boundary every cycle
        relative_radius = sweep_path(rng, len(path), settings["radial_min"], settings["radial_max"],
                                     2.2 * settings["sweeps"], dither_scale=0.1)
        taper = np.clip((relative_radius - settings["radial_bend"]) / (1.0 - settings["radial_bend"]), 0.0, 1.0)
        allowed = settings["angle_limit"] + (settings["angle_limit_taut"] - settings["angle_limit"]) * taper
        radius = cfg["length_m"] * relative_radius
        angle = np.clip(np.arctan2(path[:, 1], path[:, 0]), -allowed, allowed)
        relative_yaw = sweep_path(rng, len(path), -settings["yaw_limit"], settings["yaw_limit"],
                                  0.7 * settings["sweeps"]) * allowed / settings["angle_limit"]
        for index in range(1, len(path)):
            radial_change = radius[index] - radius[index - 1]
            angle_change = angle[index] - angle[index - 1]
            arc_bound = abs(radial_change) + max(radius[index], radius[index - 1]) * abs(angle_change)
            fraction = min(1., settings["max_speed"] * settings["control_dt"] / max(arc_bound, 1e-15))
            radius[index] = radius[index - 1] + fraction * radial_change
            angle[index] = angle[index - 1] + fraction * angle_change
        yaw = angle + relative_yaw
        for index in range(1, len(yaw)):
            yaw[index] = yaw[index - 1] + np.clip(yaw[index] - yaw[index - 1],
                                                  -settings["max_yaw_rate"] * settings["control_dt"],
                                                  settings["max_yaw_rate"] * settings["control_dt"])
        path = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), yaw))
        hold = settings["hold_control_steps"]
        if hold < 1 or 2 * hold >= len(path):
            raise ValueError("Holds must leave a nonempty dynamic excitation interval")
        path[:hold] = path[hold]
        path[-hold:] = path[-hold]
        radius = np.linalg.norm(path[:, :2], axis=1) / cfg["length_m"]
        if np.any(radius < settings["radial_min"] - 1e-12) or np.any(radius > settings["radial_max"] + 1e-12):
            raise ValueError("Excitation leaves the declared radial range")
        if np.max(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)) > (
                settings["max_speed"] * settings["control_dt"] + 1e-12):
            raise ValueError("Excitation exceeds its velocity bound")
        bearing = np.arctan2(path[:, 1], path[:, 0])
        if np.max(np.abs(path[:, 2] - bearing)) > settings["yaw_limit"] + 0.05 or np.max(
                np.abs(np.diff(path[:, 2]))) > settings["max_yaw_rate"] * settings["control_dt"] + 1e-12:
            raise ValueError("Yaw excitation leaves its declared relative range or rate")
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
        self.failure = None   # SOFA swallows exceptions raised in event handlers: record, then raise outside
        self.section_lengths = np.asarray(cable.force_field.length.value, dtype=float)
        self.samples = {key: [] for key in (
            "shapes", "commands", "gripper", "gripper_yaw", "strain", "strain_velocity",
            "tip_poses", "chain_length", "physical_length", "frames_xy", "grasp_frame_poses",
            "grasp_targets", "grasp_wrenches")}
        if getattr(cable, "modal_mo", None) is not None:
            self.samples.update(modal_positions=[], modal_velocities_sofa=[])

    def onAnimateBeginEvent(self, event):
        if self.failure is not None:
            return
        sample = min(self.step // self.substeps, len(self.path) - 1)
        point = self.path[sample]
        following = self.path[min(sample + 1, len(self.path) - 1)]
        if self.step % self.substeps == 0 and self.step // self.substeps < len(self.path):
            frames = np.asarray(self.cable.frame_poses())
            strain = self.cable.strain_mo.position.value.copy().ravel()
            grasp = self.cable.grasp_index
            target = np.asarray(self.cable.grasp_target_mo.position.value[0], dtype=float).copy()
            # reaction of the attachment at the sample state (spring law; the start-of-step
            # force SOFA stores also carries k x the command increment of that step)
            spring = self.cable.grasp_spring
            delta = pose_delta(frames[grasp], target)
            wrench = np.r_[-float(spring.stiffness.value[0]) * delta[:3],
                           -float(spring.angularStiffness.value[0]) * delta[3:]]
            values = {
                "shapes": np.asarray(self.cable.marker_positions())[:, :2].ravel(),
                "commands": (following[:2] - point[:2]) / self.control_dt,
                "gripper": point[:2].copy(),
                "gripper_yaw": point[2] if len(point) > 2 else 0.0,
                "strain": strain,
                "strain_velocity": self.cable.strain_mo.velocity.value.copy().ravel(),
                "tip_poses": frames[-1].copy(),
                # polyline chord of the frames (discretisation) vs the physical length from the strains
                "chain_length": np.linalg.norm(np.diff(frames[:, :3], axis=0), axis=1).sum(),
                "physical_length": physical_length(strain, self.section_lengths)[0],
                "frames_xy": frames[:, :2].ravel(),
                "grasp_frame_poses": frames[grasp].copy(),
                "grasp_targets": target,
                "grasp_wrenches": wrench,
            }
            if getattr(self.cable, "modal_mo", None) is not None:
                values["modal_positions"] = self.cable.modal_mo.position.value.copy().ravel()
                values["modal_velocities_sofa"] = self.cable.modal_mo.velocity.value.copy().ravel()
            if any(not np.all(np.isfinite(value)) for value in values.values()):
                self.failure = f"non-finite SOFA state at sample {sample} (t = {sample * self.control_dt:.2f} s)"
                return
            for key, value in values.items():
                self.samples[key].append(value)
        fraction = (self.step % self.substeps + 1) / self.substeps
        target = point + fraction * (following - point)
        self.coupling.update_grasp(gripper_pose(target), 1. + (self.step + 1) * self.sofa_dt)
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
    if not coupling.latched:
        raise ValueError("The SOFA training cable did not attach")
    start = np.r_[np.asarray(cable.tip_pose()[:2]), 0.0]
    goal = np.r_[np.asarray(path[0], dtype=float)[:2], path[0][2] if len(path[0]) > 2 else 0.0]
    warm_steps = max(1, math.ceil(max(np.linalg.norm(goal[:2] - start[:2]) / 0.1, abs(goal[2]) / 0.5)
                                  / cfg["timestep_s"]))
    for index in range(warm_steps + 200):
        fraction = min(1., (index + 1) / warm_steps)
        coupling.update_grasp(gripper_pose(start + fraction * (goal - start)), (index + 1) * cfg["timestep_s"])
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
            if controller.failure is not None:
                raise ValueError(f"{'ROM' if reduction is not None else 'FOM'} diverged: {controller.failure}")
        mean_step = (time.perf_counter() - started) / (len(path) * controller.substeps)
        samples = {key: np.asarray(value) for key, value in controller.samples.items()}
    finally:
        Sofa.Simulation.unload(root)
    if state_file:
        native = read_state_file(state_file, cm.strain_dimension(cfg))
        if native["times"].shape != (len(path),):
            raise ValueError(f"WriteState wrote {len(native['times'])}, expected {len(path)} samples")
        np.testing.assert_allclose(native["times"], start_time + np.arange(len(path)) * control_dt,
                       rtol=0., atol=1e-8)
        np.testing.assert_allclose(native["strain"], samples["strain"], rtol=1e-5, atol=1e-6)
        if np.any(native["reference"] != 0.):
            raise ValueError("The current cable requires the physical zero-strain reference")
        inactive = np.setdiff1d(np.arange(cm.strain_dimension(cfg)), cm.active_components(cfg))
        if cfg["planar"] and inactive.size and np.max(np.abs(native["strain"][:, inactive])) > 1e-10:
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
#!/usr/bin/env python3
"""Generate the SOFA dataset used to identify the reduced cable model.

Step 7 of the plan. For every physical-parameter vertex, the cable is excited
with safe planar gripper motions and the resulting shapes are recorded as
``(x_k, u_k, x_k+1)`` triplets.

Samples are recorded at the CONTROLLER rate with the command held across the
SOFA substeps of each period, and the period must be an exact multiple of the
SOFA timestep. Otherwise the fitted transition spans one interval while the
model claims another.

Several independent trajectories are produced per parameter vertex so the
downstream split can be made by trajectory: a rollout over scattered samples
is not a trajectory and its error is meaningless.

Runs SOFA headless, with no ROS and no Gazebo: identification data must not
depend on the rest of the stack being up.

    ros2 run cable_ts_control generate_sofa_dataset \
        --config <cable_truth.yaml> --output /tmp/cable_dataset.npz
"""

import argparse
import itertools
import os
import sys

import numpy as np


def excitation_path(rng, n_steps, dt, length, radial_min, radial_max,
                    angle_limit, max_speed, sweeps=6.0):
    """Gripper path on the annulus around the clamped base, at the control rate.

    Polar, not Cartesian: the gripper is confined to an annulus anyway (driving
    it inwards buckles the rod), and a Cartesian random walk covers the
    reachable arc very unevenly, which leaves whole TS rule cells with a
    handful of samples. Sweeping the angle deterministically and dithering it
    guarantees the shape range is covered; the dither keeps the dynamics
    excited rather than quasi-static.
    """
    step = np.arange(n_steps)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    base_angle = angle_limit * np.sin(
        2.0 * np.pi * sweeps * step / n_steps + phase)

    dither = np.zeros(n_steps)
    current = 0.0
    for k in range(n_steps):
        current = 0.85 * current + 0.15 * rng.normal(0.0, 0.35 * angle_limit)
        dither[k] = current
    angle = np.clip(base_angle + dither, -angle_limit, angle_limit)

    radius = np.zeros(n_steps)
    current = 0.5 * (radial_min + radial_max)
    for k in range(n_steps):
        current = 0.97 * current + 0.03 * rng.uniform(radial_min, radial_max)
        radius[k] = current
    radius = np.clip(radius, radial_min, radial_max) * length

    path = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])

    # Rate limit so the gripper stays inside the robot's velocity budget.
    limited = np.zeros_like(path)
    limited[0] = path[0]
    for k in range(1, n_steps):
        delta = path[k] - limited[k - 1]
        speed = np.linalg.norm(delta) / dt
        if speed > max_speed:
            delta *= max_speed / speed
        limited[k] = limited[k - 1] + delta
    return limited


def run_rollout(cfg, parameters, path, control_dt, substeps, settle_steps=50):
    """Excite one SOFA cable along a prescribed gripper path.

    One entry of ``path`` per control period; the grasp target is held while
    SOFA takes ``substeps`` internal steps, so the recorded transition really
    spans ``control_dt``.

    Returns ``(shapes, commands, gripper)`` sampled at the control rate.
    """
    import Sofa.Core
    import Sofa.Simulation
    from cable_identification import cosserat_model as cm
    from cable_identification.coupling import GraspCoupling

    sofa_dt = float(cfg["timestep_s"])
    root = Sofa.Core.Node("root")
    cm.prepare_root(root, cfg)
    cable = cm.build_cable(root, cfg)
    Sofa.Simulation.init(root)

    for name, value in parameters.items():
        cable.set_parameter(name, value)

    base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    coupling = GraspCoupling(cable, attach_mode="proximity")
    coupling.on_fixture(base)

    # Let the rod settle, then latch the tip so the gripper really drives it.
    for _ in range(settle_steps):
        Sofa.Simulation.animate(root, sofa_dt)
    tip = cable.tip_pose()
    coupling.update_grasp(tip, 0.0)
    coupling.ramp_s = 0.0

    previous = np.array(tip[:2], dtype=float)
    shapes = np.zeros((len(path), 2 * len(cable.marker_indices)))
    applied = np.zeros((len(path), 2))
    gripper = np.zeros((len(path), 2))
    for step, point in enumerate(path):
        point = np.asarray(point, dtype=float)
        applied[step] = (point - previous) / control_dt
        previous = point
        coupling.update_grasp(list(point) + [tip[2]] + tip[3:7],
                              step * control_dt + 1.0)
        for _ in range(substeps):
            Sofa.Simulation.animate(root, sofa_dt)
        markers = np.asarray(cable.marker_positions(), dtype=float)
        shapes[step] = (markers[:, :2] - np.asarray(base[:2])).reshape(-1)
        gripper[step] = point

    return shapes, applied, gripper


def parameter_vertices(bounds):
    """Cartesian product of the min/max of every uncertain parameter."""
    names = sorted(bounds)
    combinations = itertools.product(*(bounds[name] for name in names))
    return [dict(zip(names, values)) for values in combinations]


def default_bounds_file():
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory("cable_ts_control"),
                        "config", "cable_parameter_bounds.yaml")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="cable YAML config")
    parser.add_argument("--output", required=True, help="dataset .npz to write")
    parser.add_argument("--control-rate-hz", type=float, default=25.0,
                        help="controller rate; must be an exact multiple of "
                             "the SOFA timestep so the fitted transition and "
                             "the model's sample time are the same number")
    parser.add_argument("--steps", type=int, default=600,
                        help="control periods per trajectory")
    parser.add_argument("--n-trajectories", type=int, default=6,
                        help="independent trajectories per parameter vertex, so "
                             "train/validation/test split by trajectory")
    parser.add_argument("--angle-limit", type=float, default=0.55,
                        help="half-range of the gripper sweep about the base [rad]")
    parser.add_argument("--sweeps", type=float, default=6.0,
                        help="number of angular sweeps over the rollout")
    parser.add_argument("--max-speed", type=float, default=0.15,
                        help="gripper speed limit during excitation [m/s]")
    parser.add_argument("--parameter-bounds", default=None,
                        help="cable_parameter_bounds.yaml (uncertainty envelope)")
    parser.add_argument("--radial-min", type=float, default=0.92,
                        help="closest the gripper may come to the base, as a "
                             "fraction of the rod length (guards buckling)")
    parser.add_argument("--radial-max", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from cable_identification import cosserat_model as cm
    from cable_identification.parameter_bounds import load_parameter_bounds

    cfg = cm.load_config(args.config)
    sofa_dt = float(cfg["timestep_s"])
    control_dt = 1.0 / args.control_rate_hz
    substeps = int(round(control_dt / sofa_dt))
    if substeps < 1 or abs(substeps * sofa_dt - control_dt) > 1e-12:
        raise SystemExit(
            f"control period {control_dt:.6f} s is not an exact multiple of the "
            f"SOFA timestep {sofa_dt:.6f} s (nearest is {substeps} substeps = "
            f"{substeps * sofa_dt:.6f} s). Choose a control rate that divides "
            f"it, or change timestep_s in the cable config.")

    rng = np.random.default_rng(args.seed)

    # The same envelope the LMIs certify and the estimator clamps to.
    bounds_file = args.parameter_bounds or default_bounds_file()
    bounds = load_parameter_bounds(bounds_file)
    vertices = parameter_vertices(bounds)
    print(f"parameter envelope from {bounds_file}: {bounds}")
    print(f"control period {control_dt * 1e3:.3f} ms = {substeps} SOFA substeps")

    # The same trajectories for every vertex: if each got its own random
    # excitation, the fitted differences between vertices would mix the
    # parameter effect with the trajectory difference.
    paths = [excitation_path(rng, args.steps, control_dt, float(cfg["length_m"]),
                             args.radial_min, args.radial_max,
                             args.angle_limit, args.max_speed, args.sweeps)
             for _ in range(args.n_trajectories)]

    shapes, commands, grippers = [], [], []
    vertex_ids, trajectory_ids = [], []
    for index, parameters in enumerate(vertices):
        for traj, path in enumerate(paths):
            print(f"[vertex {index + 1}/{len(vertices)}] "
                  f"trajectory {traj + 1}/{len(paths)} with {parameters}")
            shape, command, gripper = run_rollout(
                cfg, parameters, path, control_dt, substeps)
            shapes.append(shape)
            commands.append(command)
            grippers.append(gripper)
            vertex_ids.append(np.full(len(shape), index))
            trajectory_ids.append(np.full(len(shape), traj))

    np.savez(
        args.output,
        shapes=np.concatenate(shapes),
        commands=np.concatenate(commands),
        gripper=np.concatenate(grippers),
        vertex_ids=np.concatenate(vertex_ids),
        trajectory_ids=np.concatenate(trajectory_ids),
        parameter_names=np.array(sorted(bounds)),
        parameter_values=np.array([[v[name] for name in sorted(bounds)]
                                   for v in vertices]),
        marker_s_over_l=np.array(cfg["marker_s_over_l"]),
        timestep_s=np.array([control_dt]),
        sofa_timestep_s=np.array([sofa_dt]),
    )
    print(f"wrote {args.output}: {sum(len(s) for s in shapes)} samples, "
          f"{len(vertices)} parameter vertices x {len(paths)} trajectories")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Generate the SOFA dataset used to identify the reduced cable model.

Step 7 of the plan. For every physical-parameter vertex, the cable is excited
with safe planar gripper motions and the resulting shapes are recorded as
``(x_k, u_k, x_k+1)`` triplets.

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


def excitation_trajectory(rng, n_steps, amplitude, max_speed, dt):
    """Band-limited random walk of the planar gripper velocity.

    A pure white command barely excites the low-frequency bending modes, so the
    command is smoothed; the displacement is clamped to keep the tip inside the
    workspace the model is meant to cover.
    """
    velocity = np.zeros((n_steps, 2))
    position = np.zeros(2)
    current = np.zeros(2)
    for step in range(n_steps):
        current = 0.9 * current + 0.1 * rng.normal(0.0, max_speed, size=2)
        current = np.clip(current, -max_speed, max_speed)
        # Steer back when the excitation drifts out of the safe box.
        for axis in range(2):
            if abs(position[axis]) > amplitude:
                current[axis] = -np.sign(position[axis]) * abs(current[axis])
        position = position + current * dt
        velocity[step] = current
    return velocity


def excitation_path(rng, n_steps, dt, length, radial_min, radial_max,
                    angle_limit, max_speed, sweeps=6.0):
    """Gripper path on the annulus around the clamped base.

    Polar, not Cartesian: the gripper is confined to an annulus anyway (driving
    it inwards buckles the rod), and a Cartesian random walk covers the
    reachable arc very unevenly, which leaves whole TS rule cells with a
    handful of samples. Sweeping the angle deterministically and dithering it
    guarantees the shape range is covered; the dither keeps the dynamics
    excited rather than quasi-static.
    """
    step = np.arange(n_steps)
    base_angle = angle_limit * np.sin(2.0 * np.pi * sweeps * step / n_steps)

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


def run_rollout(cfg, parameters, path, dt, settle_steps=50):
    """Excite one SOFA cable along a prescribed gripper path.

    Returns ``(shapes, commands)`` where ``shapes`` is ``(T, 2M)`` planar
    marker coordinates relative to the cable base, and ``commands`` is the
    velocity the cable actually received, ``(T, 2)``.
    """
    import Sofa.Core
    import Sofa.Simulation
    from cable_identification import cosserat_model as cm
    from cable_identification.coupling import GraspCoupling

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
        Sofa.Simulation.animate(root, dt)
    tip = cable.tip_pose()
    coupling.update_grasp(tip, 0.0)
    coupling.ramp_s = 0.0

    previous = np.array(tip[:2], dtype=float)
    shapes = np.zeros((len(path), 2 * len(cable.marker_indices)))
    applied = np.zeros((len(path), 2))
    for step, point in enumerate(path):
        applied[step] = (point - previous) / dt
        previous = np.asarray(point, dtype=float)
        coupling.update_grasp(list(point) + [tip[2]] + tip[3:7], step * dt + 1.0)
        Sofa.Simulation.animate(root, dt)
        markers = np.asarray(cable.marker_positions(), dtype=float)
        shapes[step] = (markers[:, :2] - np.asarray(base[:2])).reshape(-1)

    return shapes, applied


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
    parser.add_argument("--steps", type=int, default=1500,
                        help="excitation steps per parameter vertex")
    parser.add_argument("--angle-limit", type=float, default=0.55,
                        help="half-range of the gripper sweep about the base [rad]")
    parser.add_argument("--sweeps", type=float, default=6.0,
                        help="number of angular sweeps over the rollout")
    parser.add_argument("--max-speed", type=float, default=0.05,
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
    dt = float(cfg["timestep_s"])
    rng = np.random.default_rng(args.seed)

    # The same envelope the LMIs certify and the estimator clamps to.
    bounds_file = args.parameter_bounds or default_bounds_file()
    bounds = load_parameter_bounds(bounds_file)
    vertices = parameter_vertices(bounds)
    print(f"parameter envelope from {bounds_file}: {bounds}")

    shapes, commands, vertex_ids = [], [], []
    # One trajectory for every vertex: if each got its own random excitation,
    # the fitted differences between vertices would mix the parameter effect
    # with the trajectory difference, and the vertex models would not be
    # comparable.
    path = excitation_path(rng, args.steps, dt, float(cfg["length_m"]),
                           args.radial_min, args.radial_max,
                           args.angle_limit, args.max_speed, args.sweeps)
    for index, parameters in enumerate(vertices):
        print(f"[{index + 1}/{len(vertices)}] rollout with {parameters}")
        shape, command = run_rollout(cfg, parameters, path, dt)
        shapes.append(shape)
        commands.append(command)
        vertex_ids.append(np.full(len(shape), index))

    np.savez(
        args.output,
        shapes=np.concatenate(shapes),
        commands=np.concatenate(commands),
        vertex_ids=np.concatenate(vertex_ids),
        parameter_names=np.array(sorted(bounds)),
        parameter_values=np.array([[v[name] for name in sorted(bounds)]
                                   for v in vertices]),
        marker_s_over_l=np.array(cfg["marker_s_over_l"]),
        timestep_s=np.array([dt]),
    )
    print(f"wrote {args.output}: {sum(len(s) for s in shapes)} samples, "
          f"{len(vertices)} parameter vertices")
    return 0


if __name__ == "__main__":
    sys.exit(main())

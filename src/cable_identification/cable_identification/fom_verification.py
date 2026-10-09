"""Numerical verification of the planar extensible Cosserat cable (headless, FOM only).

Every check compares the SOFA model (``cosserat_model.build_cable``, Vec6d strains,
patched Cosserat plugin) with a closed form, a finite difference or a conservation law:

  1. axial traction     dL = F L0 / EA  (tip pulled through the grasp spring)
  2. pure bending       kappa = M / EI, tip on the circular arc of radius EI / M
  3. kinematics/forces  applyJ == d apply/dq (FD) at a bent + stretched state, and
                        J^T lambda == -grad V of the grasp potential (virtual work)
  4. grasp              latch, planar drag with a commanded yaw: position/orientation
                        tracking of the grasped frame, base reaction == -gripper force
  5. dynamics/energy    free oscillation: dE + h q'+^T D q'+ - W_conv = integrator dissipation
                        <= 0 every step (W_conv: discrete work of the explicit convective
                        wrench, ~0); the scene's convective wrench equals an independent
                        finite difference at the same state; its size vs f_int is reported
  6. convergence        sections, frames and timestep refined separately, on a bent hold
                        and on the taut boundary of the declared envelope (bearing and
                        relative yaw 0.2 rad under tension: the clamp boundary layer)

Numerical verification only: the nominal parameters are not calibrated on hardware.
Prints ``CABLE_FOM_TEST_PASSED`` or ``CABLE_FOM_TEST_FAILED``.
"""

import math
import sys

import numpy as np

from cable_identification.dynamics_dump import (
    NB, frame_jacobians, mapped_wrench, pose_delta, tip_jacobian_fd, tip_jacobian_sofa,
)

CONFIG = "/ros2_ws/src/cable_identification/config/cable_truth.yaml"


def build(cfg, extra_plugins=()):
    import Sofa.Core
    import Sofa.Simulation
    from cable_identification import cosserat_model as cm

    root = Sofa.Core.Node("cable_fom_test")
    cm.prepare_root(root, cfg, extra_plugins=list(extra_plugins))
    cable = cm.build_cable(root, cfg)
    return root, cable


def step(root, seconds):
    import Sofa.Simulation
    for _ in range(round(seconds / root.dt.value)):
        Sofa.Simulation.animate(root, root.dt.value)


def settle(root, cable, seconds=20.0):
    """Static equilibrium: temporary mass-proportional damping, then the physical setting."""
    import Sofa.Simulation
    physical = cable.get_parameter("rayleigh_mass")
    cable.set_parameter("rayleigh_mass", 5.0)
    step(root, seconds)
    cable.set_parameter("rayleigh_mass", physical)
    Sofa.Simulation.animate(root, root.dt.value)
    return float(np.abs(cable.strain_mo.velocity.value).max())


def energies(cable):
    """Kinetic energy of the frames (translation + rotation) and elastic energy of the strains."""
    from cable_identification import cosserat_model as cm

    cfg = cable.cfg
    mass = [float(v) for v in cm.frame_rigid_mass(cfg).split()]
    m, inertia = mass[0], mass[0] * np.array([mass[2], mass[6], mass[10]])
    frames = np.asarray(cable.frames_mo.position.value)
    velocity = np.asarray(cable.frames_mo.velocity.value)
    kinetic = 0.5 * m * np.sum(velocity[:, :3] ** 2)
    for pose, twist in zip(frames, velocity):
        x, y, z, w = pose[3:7]
        rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        body_omega = rotation.T @ twist[3:]
        kinetic += 0.5 * float(body_omega @ (inertia * body_omega))
    weights = cm.strain_weights(cfg)
    strain = np.asarray(cable.strain_mo.position.value).ravel()
    return kinetic, 0.5 * float(np.sum(weights * strain ** 2))


def grasp_potential(cable, target):
    k, k_a = float(cable.cfg["grasp_stiffness"]), float(cable.cfg["grasp_angular_stiffness"])
    delta = pose_delta(np.asarray(cable.tip_pose()), np.asarray(target))
    return 0.5 * k * float(delta[:3] @ delta[:3]) + 0.5 * k_a * float(delta[3:] @ delta[3:])


def convective_force(cable, eps=1e-6):
    """Generalized force of the mapping acceleration, -J^T m (dJ/dt q'), by FD along q'
    (independent of the scene's ConvectiveInertia controller: central difference, own eps)."""
    from cable_identification import cosserat_model as cm

    mass = [float(v) for v in cm.frame_rigid_mass(cable.cfg).split()]
    m, inertia = mass[0], mass[0] * np.array([mass[2], mass[6], mass[10]])
    saved = cable.save_state()
    q_dot = np.asarray(saved["strain_velocity"])
    velocities = []
    for sign in (1.0, -1.0):
        with cable.strain_mo.position.writeable() as x:
            x[:] = saved["strain"] + sign * eps * q_dot
        cable.refresh_mapping()
        velocities.append(np.asarray(cable.frames_mo.velocity.value).copy())
    cable.restore_state(saved)
    cable.refresh_mapping()
    acceleration = (velocities[0] - velocities[1]) / (2 * eps)
    # planar: the rotation is about z, where the world inertia equals the body I_zz
    wrench = np.c_[-m * acceleration[:, :3], -inertia[2] * acceleration[:, 3:]]
    return mapped_wrench(frame_jacobians(cable), wrench)


def check_traction(cfg, results):
    import Sofa.Simulation
    root, cable = build(cfg)
    Sofa.Simulation.init(root)
    L, EA, k, d = cfg["length_m"], cfg["EA_N"], cfg["grasp_stiffness"], 0.01
    cable.set_grasp_pose([L + d, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    settle(root, cable)
    tip = cable.tip_pose()
    force = k * (L + d - tip[0])
    expected = force * L / EA
    error = abs((tip[0] - L) / expected - 1.0)
    results.append(("axial traction dL = F L0/EA (rel. error < 1e-3)", error < 1e-3,
                    f"dL={1e3 * (tip[0] - L):.4f} mm, F={force:.3f} N, F L/EA={1e3 * expected:.4f} mm, "
                    f"rel={error:.2e}, eps_x={cable.strain_mo.position.value[:, 3].mean():.3e}"))
    Sofa.Simulation.unload(root)


def check_bending(cfg, results):
    import Sofa.Simulation
    cfg = dict(cfg, grasp_tip=False)
    root, cable = build(cfg, ["Sofa.Component.MechanicalLoad"])
    moment = 0.005
    cable.frames_mo.getContext().addObject(
        "ConstantForceField", template="Rigid3d", indices=[len(cable.frame_poses()) - 1],
        forces=[[0.0, 0.0, 0.0, 0.0, 0.0, moment]])
    Sofa.Simulation.init(root)
    settle(root, cable)
    EI, L = cfg["EI_Nm2"], cfg["length_m"]
    kappa = cable.strain_mo.position.value[:, 2]
    radius, theta = EI / moment, moment * L / EI
    tip = np.asarray(cable.tip_pose())
    arc = np.array([radius * math.sin(theta), radius * (1.0 - math.cos(theta)), 0.0])
    angle = 2.0 * math.atan2(tip[5], tip[6])
    ok = (np.allclose(kappa, moment / EI, rtol=1e-6, atol=0.0)
          and np.linalg.norm(tip[:3] - arc) < 1e-6 and abs(angle - theta) < 1e-6)
    results.append(("pure bending kappa = M/EI, tip on the circular arc", ok,
                    f"kappa={kappa.min():.6f}..{kappa.max():.6f} (M/EI={moment / EI:.6f}), "
                    f"|tip - arc|={1e3 * np.linalg.norm(tip[:3] - arc):.2e} mm, tip angle {angle:.6f} "
                    f"({theta:.6f}), |eps|max={np.abs(cable.strain_mo.position.value[:, 3:]).max():.1e}"))
    Sofa.Simulation.unload(root)


def check_jacobian(cfg, results):
    import Sofa.Simulation
    root, cable = build(cfg)
    Sofa.Simulation.init(root)
    rng = np.random.default_rng(7)
    ns = cfg["number_of_sections"]
    strains = np.zeros((ns, 6))
    strains[:, 2] = rng.normal(0.0, 1.5, ns)
    strains[:, 3] = rng.normal(0.0, 0.02, ns)
    strains[:, 4] = rng.normal(0.0, 0.02, ns)
    base = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    with cable.strain_mo.position.writeable() as x:
        x[:] = strains
    cable.refresh_mapping()
    J = tip_jacobian_sofa(cable)
    J_fd = tip_jacobian_fd(cable, base, strains, eps=1e-5)
    jac_error = np.abs(J - J_fd).max() / np.abs(J_fd).max()
    # virtual work: J^T lambda of the grasp spring must be -grad V (FD on the strains)
    target = np.asarray(cable.tip_pose()) + np.r_[0.02, -0.03, 0.0, 0.0, 0.0, 0.0, 0.0]
    target[3:] /= np.linalg.norm(target[3:])
    k, k_a = cfg["grasp_stiffness"], cfg["grasp_angular_stiffness"]
    delta = pose_delta(np.asarray(cable.tip_pose()), target)
    lam = np.r_[-k * delta[:3], -k_a * delta[3:]]
    generalized = (J.T @ lam)[NB:]
    gradient = np.zeros(strains.size)
    eps = 1e-6
    for i in range(strains.size):
        bump = np.zeros(strains.size)
        bump[i] = eps
        for sign in (1.0, -1.0):
            with cable.strain_mo.position.writeable() as x:
                x[:] = strains + sign * bump.reshape(strains.shape)
            cable.refresh_mapping()
            gradient[i] += sign * grasp_potential(cable, target) / (2 * eps)
    work_error = np.linalg.norm(generalized + gradient) / np.linalg.norm(gradient)
    results.append(("applyJ == d apply/dq (FD, rel < 1e-6) at a bent + stretched state",
                    jac_error < 1e-6, f"max|J - J_fd|/max|J| = {jac_error:.2e}"))
    results.append(("virtual work: J^T lambda == -grad V_grasp (FD, rel < 1e-5)", work_error < 1e-5,
                    f"rel = {work_error:.2e}"))
    Sofa.Simulation.unload(root)


def check_grasp(cfg, results):
    import Sofa.Simulation
    from cable_identification.coupling import GraspCoupling
    root, cable = build(cfg)
    coupling = GraspCoupling(cable, attach_mode="explicit")
    Sofa.Simulation.init(root)
    coupling.on_fixture([0.0] * 6 + [1.0])
    coupling.request_attach()
    L, h = cfg["length_m"], cfg["timestep_s"]
    latched = coupling.update_grasp([L, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.0)
    yaw = 0.5
    goal = np.array([0.95 * L * math.cos(0.3), 0.95 * L * math.sin(0.3), 0.0])
    pose = lambda p, a: [*p, 0.0, 0.0, math.sin(a / 2), math.cos(a / 2)]  # noqa: E731
    start, t = np.array([L, 0.0, 0.0]), 0.0
    n = round(3.0 / h)
    for i in range(n):
        f = (i + 1) / n
        coupling.update_grasp(pose(start + f * (goal - start), f * yaw), t)
        t += h
        Sofa.Simulation.animate(root, h)
    velocity_settle = settle(root, cable)
    frame = np.asarray(cable.frame_pose(coupling.grasp_index))
    target = np.asarray(cable.grasp_target_mo.position.value[0])
    delta = pose_delta(frame, target)
    position_error, angle_error = np.linalg.norm(delta[:3]), np.linalg.norm(delta[3:])
    frame_yaw = 2.0 * math.atan2(frame[5], frame[6])
    # static balance: the clamp reaction on the base equals minus the gripper force
    lam = np.asarray(cable.frames_mo.force.value)[coupling.grasp_index]
    base_force = np.asarray(cable.base_mo.force.value).ravel()
    J = tip_jacobian_sofa(cable)
    clamp = base_force[:3] - (J.T @ lam)[:3]
    balance = np.linalg.norm(clamp + lam[:3]) / np.linalg.norm(lam[:3])
    ok = (latched and coupling.grasp_index == len(cable.frame_poses()) - 1
          and position_error < 2e-4 and angle_error < 5e-3 and abs(frame_yaw - yaw) < 5e-3
          and balance < 1e-6 and velocity_settle < 1e-6)
    results.append(("grasp: latch at the tip, planar drag + yaw tracked, base reaction = -F",
                    ok, f"position err {1e3 * position_error:.3f} mm, orientation err "
                        f"{1e3 * angle_error:.2f} mrad, yaw {frame_yaw:.4f} ({yaw}), |F|={np.linalg.norm(lam[:3]):.3f} N, "
                        f"balance rel {balance:.1e}, settled |v|={velocity_settle:.1e}"))
    # detach: the end is free again and the cable relaxes towards its rest length
    coupling.request_detach()
    step(root, 30.0)
    frames = np.asarray(cable.frame_poses())
    chain = np.linalg.norm(np.diff(frames[:, :3], axis=0), axis=1).sum()
    results.append(("detach releases the end (free cable relaxes)", coupling.state == 0
                    and float(cable.grasp_spring.stiffness.value[0]) == 0.0 and abs(chain - L) < 2e-4,
                    f"chain {chain:.5f} m after 30 s, state {coupling.state}"))
    Sofa.Simulation.unload(root)


def free_oscillation(cfg, seconds, kappa=1.0):
    """Release a uniformly bent rod; returns the per-step energy bookkeeping and the tip trace."""
    import Sofa.Simulation
    from cable_identification import cosserat_model as cm
    cfg = dict(cfg, grasp_tip=False)
    root, cable = build(cfg)
    Sofa.Simulation.init(root)
    with cable.strain_mo.position.writeable() as x:
        x[:, 2] = kappa
    cable.refresh_mapping()
    weights = cm.strain_weights(cfg)
    damping = cable.kelvin_voigt_s * weights
    h = root.dt.value
    T0, V0 = energies(cable)
    rows, tips = [], []
    for _ in range(round(seconds / h)):
        before = cable.save_state()
        Sofa.Simulation.animate(root, h)
        q_dot = np.asarray(cable.strain_mo.velocity.value).ravel()
        T1, V1 = energies(cable)
        dissipated = h * float(np.sum(damping * q_dot ** 2))
        # discrete work of the explicit convective wrench (start-of-step value) over the step
        convective_work = h * float(np.sum(np.asarray(cable.convective_wrench)
                                           * np.asarray(cable.frames_mo.velocity.value)))
        rows.append((T1 + V1 - (T0 + V0), dissipated, convective_work))
        T0, V0 = T1, V1
        tips.append(cable.tip_pose()[:3])
    # the scene's convective wrench (computed at the start of the last step) vs an independent
    # FD at that same state
    after = cable.save_state()
    cable.restore_state(before)
    cable.refresh_mapping()
    convective = convective_force(cable)
    scene = mapped_wrench(frame_jacobians(cable), np.asarray(cable.convective_wrench))
    elastic = -weights * np.asarray(before["strain"]).ravel()
    ratio = (np.linalg.norm(convective[NB:]) / np.linalg.norm(elastic),
             np.linalg.norm(scene[NB:] - convective[NB:]) / np.linalg.norm(convective[NB:]))
    cable.restore_state(after)
    cable.refresh_mapping()
    Sofa.Simulation.unload(root)
    return np.asarray(rows), np.asarray(tips), ratio


def check_energy(cfg, results):
    # the energy balance and the wrench check exercise the optional convective term
    rows, _, ratio = free_oscillation(dict(cfg, convective_inertia=True), 4.0)
    # dE + physical dissipation - convective work = what the integrator removed
    numerical = rows[:, 0] + rows[:, 1] - rows[:, 2]
    total_kv, convective_work = rows[:, 1].sum(), rows[:, 2].sum()
    ok = np.all(numerical <= 1e-12) and total_kv > 0.0 and abs(convective_work) < 1e-2 * total_kv
    results.append(("energy: dE + h q'^T D q' - W_conv <= 0 every step (implicit Euler dissipates)", ok,
                    f"KV dissipation {total_kv:.3e} J, integrator dissipation {-numerical.sum():.3e} J "
                    f"({-numerical.sum() / total_kv:.2f} x KV), convective work {convective_work:.1e} J "
                    f"over 4 s, max positive residual {numerical.max():.1e} J"))
    results.append(("convective inertia wrench of the scene == independent FD (rel < 1e-3)",
                    ratio[1] < 1e-3, f"rel {ratio[1]:.2e}; |J^T m dJ/dt q'| / |f_int| = {ratio[0]:.2e} "
                    f"at the end of the oscillation (measured; the plant runs with convective_inertia="
                    f"{cfg.get('convective_inertia', False)})"))


def check_convergence(cfg, results):
    import Sofa.Simulation
    from cable_identification.coupling import GraspCoupling
    L = cfg["length_m"]

    def static_markers(overrides, radius, bearing, relative_yaw):
        c = dict(cfg, **overrides)
        root, cable = build(c)
        coupling = GraspCoupling(cable, attach_mode="explicit")
        Sofa.Simulation.init(root)
        coupling.on_fixture([0.0] * 6 + [1.0])
        coupling.request_attach()
        coupling.update_grasp([L, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.0)
        goal, yaw = radius * L * np.array([math.cos(bearing), math.sin(bearing)]), bearing + relative_yaw
        h, n = c["timestep_s"], round(4.0 / c["timestep_s"])
        for i in range(n):
            f = (i + 1) / n
            p, a = np.array([L, 0.0]) + f * (goal - np.array([L, 0.0])), f * yaw
            coupling.update_grasp([*p, 0.0, 0.0, 0.0, math.sin(a / 2), math.cos(a / 2)], i * h)
            Sofa.Simulation.animate(root, h)
        settle(root, cable)
        markers = np.asarray(cable.marker_positions())[:, :2]
        kappa = float(np.abs(cable.strain_mo.position.value[:, 2]).max())
        tension = float(cfg["EA_N"] * cable.strain_mo.position.value[:, 3].mean())
        Sofa.Simulation.unload(root)
        return markers, kappa, tension

    budget = 1e-3
    for label, key, values in (("sections", "number_of_sections", (8, 16, 32)),
                               ("frames", "number_of_frames", (20, 40, 80))):
        markers = [static_markers({key: v}, 0.95, 0.3, 0.0)[0] for v in values]
        steps = [np.abs(markers[i + 1] - markers[i]).max() for i in range(len(values) - 1)]
        results.append((f"convergence in {label} {values}, bent hold: last refinement moves markers < 1 mm",
                        steps[-1] < budget, "max marker change per refinement: "
                        + ", ".join(f"{1e3 * s:.3f} mm" for s in steps)))
    # taut boundary of the declared envelope (softest EI of the identification box): the
    # clamp boundary layer sqrt(EI/T) is under-resolved in curvature, the markers must not be
    taut = [static_markers({"number_of_sections": ns, "EI_Nm2": 0.005}, 1.002, 0.2, 0.2) for ns in (16, 32)]
    change = np.abs(taut[1][0] - taut[0][0]).max()
    results.append(("convergence in sections (16, 32), taut at bearing/relative yaw 0.2 rad, EI 0.005: "
                    "markers move < 1 mm", change < budget,
                    f"marker change {1e3 * change:.3f} mm; tension {taut[0][2]:.1f} N; peak |kappa| "
                    f"{taut[0][1]:.1f} -> {taut[1][1]:.1f} /m (boundary layer, not converged: reported)"))
    tips = [free_oscillation(dict(cfg, timestep_s=dt), 2.0)[1][-1] for dt in (0.02, 0.01, 0.005)]
    steps = [np.linalg.norm(tips[i + 1] - tips[i]) for i in range(2)]
    results.append(("convergence in timestep (0.02, 0.01, 0.005): tip after 2 s of free oscillation",
                    steps[-1] < 5e-3, "tip change per halving: "
                    + ", ".join(f"{1e3 * s:.2f} mm" for s in steps)
                    + f" (ratio {steps[0] / max(steps[1], 1e-12):.1f}, first order ~2)"))


def main(argv=None):
    import argparse
    from cable_identification import cosserat_model as cm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args(argv)
    cfg = cm.load_config(args.config)
    results = []
    for check in (check_traction, check_bending, check_jacobian, check_grasp, check_energy,
                  check_convergence):
        check(cfg, results)
    print()
    failed = 0
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failed += 0 if ok else 1
    print()
    if failed:
        print(f"CABLE_FOM_TEST_FAILED ({failed} failures)")
        return 1
    print("CABLE_FOM_TEST_PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Numerical verification of the planar extensible Cosserat cable (headless, FOM only).

Every check compares the SOFA model (``cosserat_model.build_cable``, Vec6d strains,
patched Cosserat plugin) with a closed form, a finite difference or a conservation law:

  1. axial traction     dL = F L0 / EA with F = lambda / h of the end attachment, at h and h/2
  2. pure bending       kappa = M / EI, tip on the circular arc of radius EI / M
  3. kinematics/forces  at zero, signed near-zero (1e-12..1e-3), bent, 3D and series/closed-form
                        switch (|kappa| l = 2) curvatures with extension and shear: apply ==
                        independent expm chain, applyJ == d apply/dq (FD, dimensionless
                        scaling) and J^T lambda == -grad V of an end-spring potential
     transpose          the force SOFA actually maps (ConstantForceField on the last frame ->
                        applyJT -> base and strain force buffers) == J^T wrench, per load
     restore            save -> perturb -> restore bitwise exact; repeated and failing probes
                        leave no drift
     fixture            fixed base and planarity over the grasp episode; placement semantics
  4. grasp              Lagrange attachment of s = L (FOM and full-space ROM): residuals,
                        identity, actual rows vs FD (H_a == H_q Phi), W (q - q0) == H^T F,
                        rejection, release, FOM == ROM
     checkpoint         save -> advance -> restore -> replay of a moving-boundary rollout
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
    NB, frame_jacobians, mapped_wrench, pose_delta, quat, tip_jacobian_fd, tip_jacobian_sofa,
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


def convective_force(cable, eps=1e-6):
    """Generalized force of the mapping acceleration, -J^T m (dJ/dt q'), by FD along q'
    (independent of the scene's ConvectiveInertia controller: central difference, own eps)."""
    from cable_identification import cosserat_model as cm

    mass = [float(v) for v in cm.frame_rigid_mass(cable.cfg).split()]
    m, inertia = mass[0], mass[0] * np.array([mass[2], mass[6], mass[10]])
    saved = cable.save_state()
    q_dot = np.array(saved["strain_velocity"], copy=True)
    velocities = []
    try:
        for sign in (1.0, -1.0):
            with cable.strain_mo.position.writeable() as x:
                x[:] = saved["strain"] + sign * eps * q_dot
            cable.refresh_mapping()
            velocities.append(np.asarray(cable.frames_mo.velocity.value).copy())
    finally:
        cable.restore_state(saved)
    acceleration = (velocities[0] - velocities[1]) / (2 * eps)
    # planar: the rotation is about z, where the world inertia equals the body I_zz
    wrench = np.c_[-m * acceleration[:, :3], -inertia[2] * acceleration[:, 3:]]
    return mapped_wrench(frame_jacobians(cable), wrench)


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


def special_states(cfg):
    """Named strain states (ns x 6) exercising every branch of the PCS kinematics, all with
    extension and in-plane shear: zero curvature, signed near-zero curvatures down to 1e-12,
    a bent rod, the series/closed-form switch |kappa| l = 2 on both sides, and a 3D state."""
    ns = int(cfg["number_of_sections"])
    switch = 2.0 / (float(cfg["length_m"]) / ns)
    rng = np.random.default_rng(7)
    stretched = np.zeros((ns, 6))
    stretched[:, 3] = rng.normal(0.0, 0.02, ns)
    stretched[:, 4] = rng.normal(0.0, 0.02, ns)
    near_zero = np.resize([0.0, 1e-12, -1e-12, 1e-9, -1e-9, 1e-6, -1e-6, 1e-3, -1e-3], ns)
    states = {"zero curvature": stretched.copy()}
    states["near-zero curvature (+/-1e-12 .. 1e-3)"] = stretched.copy()
    states["near-zero curvature (+/-1e-12 .. 1e-3)"][:, 2] = near_zero
    states["bent + stretched"] = stretched.copy()
    states["bent + stretched"][:, 2] = rng.normal(0.0, 1.5, ns)
    transition = states["bent + stretched"].copy()
    transition[:4, 2] = switch * np.array([1 - 1e-9, -(1 + 1e-9), 1 + 1e-12, -(1 - 1e-12)])
    states["series/closed-form switch |kappa| l = 2"] = transition
    spatial = states["near-zero curvature (+/-1e-12 .. 1e-3)"].copy()
    spatial[:, 0] = np.roll(near_zero, 1)
    spatial[:, 1] = np.roll(near_zero, 2)
    spatial[:, 5] = rng.normal(0.0, 0.01, ns)
    states["3D near-zero curvature + shear"] = spatial
    return states


def reference_frames(cfg, base7, strains):
    """Frames of the PCS kinematics evaluated independently of the plugin:
    g(s) = g_base prod_j expm(l_j xi_j^) expm(x xi_k^), xi = (kappa, 1 + eps_x, eps_y, eps_z)."""
    from scipy.linalg import expm
    from scipy.spatial.transform import Rotation
    from cable_identification import cosserat_model as cm

    _, lengths, boundaries, _, abscissae = cm.build_geometry(cfg)

    def twist(q):
        xi = np.zeros((4, 4))
        kx, ky, kz = q[:3]
        xi[:3, :3] = [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]]
        xi[:3, 3] = [1.0 + q[3], q[4], q[5]]
        return xi

    g0 = np.eye(4)
    g0[:3, :3] = Rotation.from_quat(base7[3:7]).as_matrix()
    g0[:3, 3] = base7[:3]
    nodes = [g0]
    for q, length in zip(strains, lengths):
        nodes.append(nodes[-1] @ expm(length * twist(q)))
    frames = []
    for s in abscissae:
        k = min(int(np.searchsorted(boundaries, s, side="right")) - 1, len(lengths) - 1)
        g = nodes[k] @ expm((s - boundaries[k]) * twist(strains[k]))
        frames.append(np.r_[g[:3, 3], Rotation.from_matrix(g[:3, :3]).as_quat()])
    return np.array(frames)


def scaled_jacobian_error(J, J_ref, length):
    """max |J - J_ref| / max |J_ref| after making every row and column dimensionless:
    rows (tip position / L, rotation [rad]); columns (base translation x L, base rotation
    [rad], curvature x 1/L, extension/shear [-])."""
    rows = np.r_[np.full(3, 1.0 / length), np.ones(3)]
    columns = np.r_[np.full(3, length), np.ones(3),
                    np.tile([1.0 / length] * 3 + [1.0] * 3, (J.shape[1] - NB) // 6)]
    J, J_ref = rows[:, None] * J * columns, rows[:, None] * J_ref * columns
    return np.abs(J - J_ref).max() / np.abs(J_ref).max()


def check_kinematics(cfg, results):
    """apply vs independent expm kinematics, applyJ vs d apply/dq and virtual work, at the
    special states (cosserat-patches/0003: z = l |kappa| branch, full translational strain)."""
    import Sofa.Simulation
    root, cable = build(cfg)
    Sofa.Simulation.init(root)
    base = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    L = float(cfg["length_m"])
    k, k_a = 1.0e5, 10.0  # test potential of an end spring (any stiffness: the identity is linear)
    for label, strains in special_states(cfg).items():
        with cable.strain_mo.position.writeable() as x:
            x[:] = strains
        cable.refresh_mapping()
        frames = np.asarray(cable.frame_poses())
        expected = reference_frames(cfg, base, strains)
        deltas = np.array([pose_delta(f, e) for f, e in zip(frames, expected)])
        position_error, angle_error = np.abs(deltas[:, :3]).max(), np.abs(deltas[:, 3:]).max()
        results.append((f"apply == independent expm kinematics [{label}] (< 1e-12 m, rad)",
                        position_error < 1e-12 and angle_error < 1e-12,
                        f"max |dp| {position_error:.1e} m, max |dtheta| {angle_error:.1e} rad"))
        J = tip_jacobian_sofa(cable)
        jac_error = scaled_jacobian_error(J, tip_jacobian_fd(cable, base, strains), L)
        target = np.asarray(cable.tip_pose()) + np.r_[0.02, -0.03, 0.0, 0.0, 0.0, 0.0, 0.0]
        target[3:] /= np.linalg.norm(target[3:])

        def potential():
            delta = pose_delta(np.asarray(cable.tip_pose()), target)
            return 0.5 * k * float(delta[:3] @ delta[:3]) + 0.5 * k_a * float(delta[3:] @ delta[3:])

        delta = pose_delta(np.asarray(cable.tip_pose()), target)
        generalized = (J.T @ np.r_[-k * delta[:3], -k_a * delta[3:]])[NB:]
        gradient = np.zeros(strains.size)
        saved = cable.save_state()
        try:
            for i in range(strains.size):
                bump = np.zeros(strains.size)
                bump[i] = 1e-6
                for sign in (1.0, -1.0):
                    with cable.strain_mo.position.writeable() as x:
                        x[:] = strains + sign * bump.reshape(strains.shape)
                    cable.refresh_mapping()
                    gradient[i] += sign * potential() / 2e-6
        finally:
            cable.restore_state(saved)
        work_error = np.linalg.norm(generalized + gradient) / np.linalg.norm(gradient)
        results.append((f"applyJ == d apply/dq (FD, scaled rel < 1e-6) [{label}]", jac_error < 1e-6,
                        f"scaled max|J - J_fd|/max|J| = {jac_error:.2e}"))
        results.append((f"virtual work J^T lambda == -grad V_end (FD, rel < 1e-5) [{label}]",
                        work_error < 1e-5, f"rel = {work_error:.2e}"))
    Sofa.Simulation.unload(root)


def check_transpose(cfg, results):
    """SOFA's actual force path (ConstantForceField on the last frame -> mapping applyJT ->
    base_mo.force, strain_mo.force) == J^T wrench with J from applyJ, at the special states.

    Isolated 3D test scene (no grasp, no planarity, zero gravity, no convective term), not a
    validation of the production dynamics. One step from rest: the force buffers hold the
    forces of the step's starting state, where J is taken.
    """
    import Sofa.Simulation
    from cable_identification import cosserat_model as cm

    local = dict(cfg, grasp_tip=False, planar=False, gravity=[0.0, 0.0, 0.0],
                 convective_inertia=False)
    L, ns = float(local["length_m"]), int(local["number_of_sections"])
    scale = np.r_[np.full(3, L), np.ones(3), np.tile([1 / L, 1 / L, 1 / L, 1, 1, 1], ns)]
    weights = cm.strain_weights(local)
    root, cable = build(local)
    last = len(cable.frame_poses()) - 1
    load = cable.frames_mo.getContext().addObject(
        "ConstantForceField", template="Rigid3d", name="transposeLoad", indices=[last],
        forces=[[0.0] * 6])
    Sofa.Simulation.init(root)
    anchor = np.array(cable.base_mo.rest_position.value[0], dtype=float, copy=True)
    loads = {"Fx 0.1 N": np.r_[0.1, 0, 0, 0, 0, 0], "Fy 0.1 N": np.r_[0, 0.1, 0, 0, 0, 0],
             "Mz 0.001 N.m": np.r_[0, 0, 0, 0, 0, 0.001]}
    report = {name: [] for name in loads}
    for label, strains in special_states(local).items():
        for name, wrench in loads.items():
            with cable.base_mo.position.writeable() as p:
                p[0] = anchor
            for data in (cable.base_mo.velocity, cable.strain_mo.velocity):
                with data.writeable() as v:
                    v[:] = 0.0
            with cable.strain_mo.position.writeable() as x:
                x[:] = strains
            cable.refresh_mapping()
            start = cable.save_state()
            J = tip_jacobian_sofa(cable)
            load.forces.value = [wrench.tolist()]
            try:
                Sofa.Simulation.animate(root, root.dt.value)
                base_force = np.array(cable.base_mo.force.value, dtype=float, copy=True).ravel()
                strain_force = np.array(cable.strain_mo.force.value, dtype=float, copy=True).ravel()
                frame_force = np.array(cable.frames_mo.force.value, dtype=float, copy=True)[last]
            finally:
                load.forces.value = [[0.0] * 6]
                cable.restore_state(start)
            hooke = -weights * (start["strain"].ravel() - start["strain_rest"].ravel())
            actual = scale * np.r_[base_force, strain_force - hooke]
            expected = scale * (J.T @ wrench)
            error = np.abs(actual - expected).max()
            limit = 1e-12 + 1e-9 * np.abs(expected).max()
            finite = np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
            loaded = np.array_equal(frame_force, wrench)
            report[name].append((label, finite and loaded and error <= limit, error, limit))
    for name, rows in report.items():
        worst = max(rows, key=lambda row: row[2] / row[3])
        results.append((f"applyJT force path == J^T wrench, {name} at the last frame, 5 special "
                        "states (scaled, <= 1e-12 + 1e-9 max|J^T w|)", all(row[1] for row in rows),
                        "; ".join(f"{label}: {error:.1e} (lim {limit:.1e})"
                                  for label, _, error, limit in rows)
                        + f"; worst {worst[2] / worst[3]:.1e} of its limit, frame load == wrench, "
                          "finite"))
    Sofa.Simulation.unload(root)


def mechanical_fields(cable):
    """Every independent field save_state covers plus the mapped frames (for bitwise checks)."""
    fields = {key: np.array(value, copy=True) for key, value in cable.save_state().items()}
    fields["frames"] = np.array(cable.frames_mo.position.value, copy=True)
    fields["frame_velocities"] = np.array(cable.frames_mo.velocity.value, copy=True)
    return fields


def round_trip(cable, rng):
    """Fields that differ after save -> perturb every independent field -> restore."""
    reference = mechanical_fields(cable)
    saved = cable.save_state()
    independent = cable.modal_mo if cable.modal_mo is not None else cable.strain_mo
    for data in (independent.position, independent.velocity, cable.strain_mo.rest_position,
                 cable.base_mo.position, cable.base_mo.velocity, cable.base_mo.rest_position):
        with data.writeable() as values:
            values[:] += rng.normal(0.0, 1e-3, values.shape)
    cable.refresh_mapping()
    cable.restore_state(saved)
    after = mechanical_fields(cable)
    return [key for key in reference if not np.array_equal(reference[key], after[key])]


def check_restore(cfg, results):
    """save -> perturb -> restore is exact (FOM and ROM), probes leave no trace (also when
    they raise)."""
    import tempfile
    import Sofa.Core
    import Sofa.Simulation
    from cable_identification import cosserat_model as cm
    from cable_identification.strain_basis import write_full_space_basis

    root, cable = build(cfg)
    Sofa.Simulation.init(root)
    rng = np.random.default_rng(3)
    with cable.strain_mo.position.writeable() as x:
        x[:] = special_states(cfg)["bent + stretched"]
    with cable.strain_mo.velocity.writeable() as v:
        v[:] = rng.normal(0.0, 0.1, v.shape)
    cable.refresh_mapping()
    reference = mechanical_fields(cable)
    mismatch = round_trip(cable, rng)
    results.append(("FOM save -> perturb all fields -> restore is bitwise exact (incl. mapped frames)",
                    not mismatch, f"fields differing: {mismatch or 'none'}"))
    drift = []
    for repeat in range(3):
        tip_jacobian_fd(cable, cable.base_mo.position.value[0], cable.strain_mo.position.value)
        frame_jacobians(cable)
        convective_force(cable)
        drift += [f"{key}@{repeat}" for key, value in mechanical_fields(cable).items()
                  if not np.array_equal(value, reference[key])]
    raised = False
    try:  # a probe that fails half way (wrong strain shape after the base was perturbed)
        tip_jacobian_fd(cable, cable.base_mo.position.value[0] + 1e-3, np.zeros(5))
    except ValueError:
        raised = True
    drift += [f"{key}@raise" for key, value in mechanical_fields(cable).items()
              if not np.array_equal(value, reference[key])]
    results.append(("3x repeated FD/applyJ/convective probes and a failing probe leave no drift",
                    raised and not drift, f"raised={raised}, drifted: {drift or 'none'}"))
    Sofa.Simulation.unload(root)
    with tempfile.TemporaryDirectory(prefix="cable-full-space-") as directory:
        root = Sofa.Core.Node("cable_rom_restore")
        cm.prepare_root(root, cfg, extra_plugins=["ModelOrderReduction"])
        cable = cm.build_cable(root, cfg, reduction=write_full_space_basis(cfg, directory))
        Sofa.Simulation.init(root)
        for data in (cable.modal_mo.position, cable.modal_mo.velocity):
            with data.writeable() as values:
                values[:] = rng.normal(0.0, 0.01, values.shape)
        cable.refresh_mapping()
        mismatch = round_trip(cable, rng)
        results.append(("ROM (full-space basis) save -> perturb -> restore is bitwise exact",
                        not mismatch, f"fields differing: {mismatch or 'none'}"))
        Sofa.Simulation.unload(root)


def check_traction(cfg, results):
    """Straight extension 0.7000 -> 0.7014 m held by the attachment: the measured reaction
    lambda / h obeys dL = F L / EA (actual extension), at h and h / 2 (a wrong time-step factor
    in the multiplier interpretation would show as a factor 2)."""
    import Sofa.Simulation
    L = float(cfg["length_m"])
    for dt in (STAGE_B_DT, STAGE_B_DT / 2):
        c = nominal_attachment_cfg(cfg, dt)
        root, cable, coupling = attached_scene(c, IDENTITY)
        command = PlanarCommand(IDENTITY, [L, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dt)
        coupling.request_attach()
        latched = coupling.update_grasp(*command.at(0, 0.0))
        command.move(0.0014, 0.0, 0.0)
        command.hold(1.0)
        for k in range(command.steps):
            coupling.update_grasp(*command.at(k + 1, (k + 1) * dt))
            Sofa.Simulation.animate(root, dt)
        raw = np.asarray(cable.attachment.findData("lambda").value, dtype=float)
        force = cable.attachment_reaction()
        extension = cable.tip_pose()[0] - L
        expected = force[0] * L / float(c["EA_N"])
        error = abs(extension - expected) / abs(expected)
        results.append((f"traction dL = F L/EA from the attachment reaction, h = {dt} s (rel <= 1e-3)",
                        latched and error <= 1e-3,
                        f"raw lambda = [{raw[0]:.6e}, {raw[1]:.1e}, {raw[2]:.1e}] N.s, F = lambda/h = "
                        f"[{force[0]:.6f} N, {force[1]:.1e} N, {force[2]:.1e} N.m]; dL {1e3 * extension:.6f} mm"
                        f" vs F L/EA {1e3 * expected:.6f} mm, rel {error:.2e}"))
        Sofa.Simulation.unload(root)


# Stage B: the approved nominal cable. Only the step is overridden (the production config
# still says timestep_s 0.01; the step is a Stage C decision).
NOMINAL = {"length_m": 0.7, "radius_m": 0.004, "mass_kg": 0.07, "EA_N": 5000.0, "EI_Nm2": 0.01,
           "GA_N": 2000.0, "GJ_Nm2": 0.008, "rayleigh_stiffness_s": 0.02, "rayleigh_mass_per_s": 0.0,
           "gravity": [0.0, 0.0, 0.0], "number_of_sections": 16, "number_of_frames": 40,
           "planar": True, "grasp_tip": True}
STAGE_B_DT = 0.0025
# Rate limits of the plan's simulation experiment (not a measured robot envelope)
COMMAND_LIMITS = {"speed": 0.030, "accel": 0.060, "yaw_rate": 0.150, "yaw_accel": 0.300}
IDENTITY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
# A tilted, shifted fixture: every plane axis is exercised (zero gravity, so legal)
FIXTURE = [0.1, -0.05, 0.02, *quat(np.array([0.3, -0.2, 0.5]))]
# Fingers-down gripper 5.4 mm from the straight end (fixture coordinates): latch offset + lever
GRIPPER_LATCH = [0.703, 0.002, 0.004, 1.0, 0.0, 0.0, 0.0]
# Declared limits (acceptance table of the Stage B instructions; solver-level for the replay)
LIMITS = {"base_position": 1e-9, "base_angle": 1e-9, "plane": 1e-8, "end_position": 1e-5,
          "end_yaw": 1e-4, "rows_fd": 1e-6, "rows_rom": 1e-12, "balance": 1e-3,
          "rom_frames": 1e-8, "rom_reaction": 1e-6, "replay_position": 1e-12, "replay_reaction": 1e-12}


def nominal_attachment_cfg(cfg, dt=STAGE_B_DT):
    differing = {key: (cfg.get(key), value) for key, value in NOMINAL.items() if cfg.get(key) != value}
    if differing:
        raise ValueError(f"Stage B checks run on the approved nominal cable; config differs: {differing}")
    return dict(cfg, timestep_s=dt)


def compose(a7, b7):
    from cable_identification.coupling import _q_mul, _q_rot
    p = np.asarray(a7[:3]) + np.asarray(_q_rot(tuple(a7[3:7]), tuple(b7[:3])))
    return [*p, *_q_mul(tuple(a7[3:7]), tuple(b7[3:7]))]


class PlanarCommand:
    """Gripper commands in the fixture plane about a latch pose (fixture coordinates):
    rest-to-rest quintic segments (dx, dy, dyaw), each as short as COMMAND_LIMITS allow.
    ``at(k, t)`` returns (pose7, time, twist6) in world coordinates for update_grasp."""

    def __init__(self, fixture7, latch_local7, h):
        self.fixture, self.latch, self.h = list(fixture7), list(latch_local7), float(h)
        self.segments, self.end, self.steps = [], np.zeros(3), 0

    def move(self, dx, dy, dyaw):
        delta, peak = np.array([dx, dy, dyaw], dtype=float), 10.0 / math.sqrt(3.0)
        D, Y = math.hypot(dx, dy), abs(dyaw)
        T = max(1.875 * D / COMMAND_LIMITS["speed"], math.sqrt(peak * D / COMMAND_LIMITS["accel"]),
                1.875 * Y / COMMAND_LIMITS["yaw_rate"], math.sqrt(peak * Y / COMMAND_LIMITS["yaw_accel"]))
        n = math.ceil(T / self.h - 1e-9)
        self.segments.append((self.steps, self.steps + n, self.end.copy(), self.end + delta))
        self.end, self.steps = self.end + delta, self.steps + n

    def hold(self, seconds):
        self.steps += round(seconds / self.h)

    def at(self, k, t):
        from cable_identification.coupling import _q_mul, _q_rot
        u, du = np.zeros(3), np.zeros(3)
        for k0, k1, a, b in self.segments:
            if k >= k1:
                u = b
                continue
            if k > k0:
                s, T = (k - k0) / (k1 - k0), (k1 - k0) * self.h
                u = a + (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5) * (b - a)
                du = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T * (b - a)
            break
        local = [self.latch[0] + u[0], self.latch[1] + u[1], self.latch[2],
                 *_q_mul((0.0, 0.0, math.sin(u[2] / 2), math.cos(u[2] / 2)), tuple(self.latch[3:7]))]
        q = tuple(self.fixture[3:7])
        twist = np.r_[_q_rot(q, (du[0], du[1], 0.0)), _q_rot(q, (0.0, 0.0, du[2]))]
        return compose(self.fixture, local), t, twist


def attached_scene(cfg, fixture, reduction=None):
    """Planar attachment scene (FOM, or ROM on ``reduction``), placed on the fixture post-init."""
    import Sofa.Core
    import Sofa.Simulation
    from cable_identification import cosserat_model as cm
    from cable_identification.coupling import GraspCoupling
    root = Sofa.Core.Node("cable_stage_b")
    cm.prepare_root(root, cfg, extra_plugins=["ModelOrderReduction"] if reduction else [])
    cable = cm.build_cable(root, cfg, reduction=reduction)
    coupling = GraspCoupling(cable, attach_mode="explicit")
    Sofa.Simulation.init(root)
    coupling.on_fixture(fixture)
    return root, cable, coupling


def constraint_rows(mo, n_rows):
    """Constraint matrix SOFA propagated to ``mo`` in the last step, dense (rows x dofs), nnz."""
    matrix = mo.constraint.value  # scipy CSR: rows = constraint ids, columns = scalar dofs
    dense = np.zeros((n_rows, mo.position.value.size))
    if matrix.shape[0] and matrix.shape[1]:
        block = matrix.toarray()
        dense[:block.shape[0], :block.shape[1]] = block
    return dense, int(matrix.nnz)


def residual_jacobian_fd(cable, independent, eps):
    """dC/dx of the attachment residual (target fixed) by central differences over every
    flat coordinate of ``independent`` with nonzero ``eps``, at the current state."""
    saved = cable.save_state()
    start = np.array(independent.position.value, dtype=float, copy=True)
    H = np.zeros((3, start.size))
    try:
        for j in np.flatnonzero(eps):
            for sign in (1.0, -1.0):
                x = start.copy()
                x.reshape(-1)[j] += sign * eps[j]
                with independent.position.writeable() as p:
                    p[:] = x
                cable.refresh_mapping()
                H[:, j] += sign * cable.attachment_residual() / (2 * eps[j])
    finally:
        cable.restore_state(saved)
    return H


def scaled_rows_error(H, H_ref, row_scale, column_scale):
    H, H_ref = row_scale[:, None] * H * column_scale, row_scale[:, None] * H_ref * column_scale
    return np.abs(H - H_ref).max() / np.abs(H_ref).max(), np.linalg.svd(H_ref, compute_uv=False)


def grasp_episode(cfg, reduction=None, phi=None):
    """Latch, translation + yaw, hold, yaw only, hold, release, free: per-step boundary
    measurements, the actual attachment rows vs FD at both holds, release bookkeeping."""
    import Sofa.Simulation
    from cable_identification import cosserat_model as cm
    from cable_identification.coupling import IncompatibleGraspCommand, _q_rot
    h = float(cfg["timestep_s"])
    L = float(cfg["length_m"])
    root, cable, coupling = attached_scene(cfg, FIXTURE, reduction)
    normal = np.asarray(_q_rot(tuple(FIXTURE[3:7]), (0.0, 0.0, 1.0)))
    out = {"latch_rejected_far": False, "rows": [], "balance": [], "steps": 0}
    command = PlanarCommand(FIXTURE, GRIPPER_LATCH, h)
    coupling.request_attach()
    far = compose(FIXTURE, [0.712, 0.002, 0.004, 1.0, 0.0, 0.0, 0.0])   # 12.8 mm from the end
    out["latch_rejected_far"] = not coupling.update_grasp(far, 0.0, np.zeros(6))
    out["latched"] = coupling.update_grasp(*command.at(0, 0.0))
    latch_offset = coupling.offset
    last = len(cable.frames_mo.position.value) - 1
    command.move(-0.04, 0.05, 0.3)
    hold_1 = command.steps + round(3.0 / h)
    command.hold(3.0)
    command.move(0.0, 0.0, -0.4)
    hold_2 = command.steps + round(3.0 / h)
    command.hold(3.0)
    independent = cable.modal_mo if cable.modal_mo is not None else cable.strain_mo
    if reduction is None:
        eps = np.zeros(independent.position.value.size)
        eps[cm.active_components(cfg)] = 1e-6
    else:
        eps = 1e-6 / np.abs(phi).max(axis=0)
    columns = np.tile([1.0 / L] * 3 + [1.0] * 3, int(cfg["number_of_sections"]))
    frames, reactions, base, plane, end, identity = [], [], [], [], [], True
    for k in range(command.steps):
        if k + 1 in (hold_1, hold_2):
            pre = cable.save_state()
        coupling.update_grasp(*command.at(k + 1, (k + 1) * h))
        Sofa.Simulation.animate(root, h)
        poses = np.array(cable.frames_mo.position.value, dtype=float)
        frames.append(poses[:, :3].copy())
        reactions.append(cable.attachment_reaction())
        delta = pose_delta(np.asarray(cable.base_mo.position.value[0], dtype=float), np.asarray(FIXTURE))
        base.append((np.linalg.norm(delta[:3]), np.linalg.norm(delta[3:])))
        plane.append(np.abs((poses[:, :3] - FIXTURE[:3]) @ normal).max())
        end.append(cable.attachment_residual())
        identity &= (int(cable.attachment.index.value) == last and coupling.offset == latch_offset
                     and coupling.latched)
        if k + 1 in (hold_1, hold_2):
            post = cable.save_state()
            H, nnz = constraint_rows(independent, 3)
            cable.restore_state(pre)
            H_fd = residual_jacobian_fd(cable, independent, eps)
            cable.restore_state(post)
            rows = np.array([1.0 / L, 1.0 / L, 1.0])
            if reduction is None:
                active = cm.active_components(cfg)
                error, sigma = scaled_rows_error(H[:, active], H_fd[:, active], rows, columns[active])
                force = cable.attachment_reaction()
                strain = np.asarray(cable.strain_mo.position.value, dtype=float).ravel()
                rest = np.asarray(cable.strain_mo.rest_position.value, dtype=float).ravel()
                residual = cm.strain_weights(cfg) * (strain - rest) - H.T @ force
                elastic = cm.strain_weights(cfg) * (strain - rest)
                moments = [i for i in active if i % 6 < 3]
                forces = [i for i in active if i % 6 >= 3]
                out["balance"].append((np.abs(residual[moments]).max() / np.abs(elastic[moments]).max(),
                                       np.abs(residual[forces]).max() / np.abs(elastic[forces]).max(),
                                       np.abs(elastic[moments]).max(), np.abs(elastic[forces]).max()))
                out["rows"].append((error, sigma, nnz, None))
            else:
                error, sigma = scaled_rows_error(H, H_fd, rows, np.ones(H.shape[1]))
                H_q, _ = constraint_rows(cable.strain_mo, 3)
                rom = np.abs(H - H_q @ phi).max() / np.abs(H).max()
                out["rows"].append((error, sigma, nnz, rom))
    out["steps"] = command.steps
    # an out-of-plane command is refused and leaves the target untouched
    target = cable.attachment_target()
    lifted = list(command.at(command.steps, 0.0)[0])
    lifted[:3] = list(np.asarray(lifted[:3]) + 0.02 * normal)
    try:
        coupling.update_grasp(lifted, command.steps * h + h, np.zeros(6))
        out["rejected_lift"] = False
    except IncompatibleGraspCommand:
        out["rejected_lift"] = np.array_equal(target, cable.attachment_target())
    # release: rows off, nothing reset, then free motion without any attachment force
    before = mechanical_fields(cable)
    coupling.request_detach()
    after = mechanical_fields(cable)
    out["release_untouched"] = all(np.array_equal(before[key], after[key]) for key in before)
    free_lambda, free_nnz = 0.0, 0
    for _ in range(round(0.5 / h)):
        Sofa.Simulation.animate(root, h)
        free_lambda = max(free_lambda, np.abs(cable.attachment.findData("lambda").value).max())
        free_nnz = max(free_nnz, constraint_rows(independent, 3)[1])
    out.update(frames=np.array(frames), reactions=np.array(reactions), base=np.array(base),
               plane=np.array(plane), end=np.array(end), identity=identity,
               free_lambda=free_lambda, free_nnz=free_nnz, active_after=cable.attachment_active())
    Sofa.Simulation.unload(root)
    return out


_EPISODES = {}


def stage_b_episodes(cfg):
    """The grasp episode on the FOM and on the full-space diagnostic ROM (run once)."""
    import tempfile
    from cable_identification.strain_basis import read_modes, write_full_space_basis
    if "fom" not in _EPISODES:
        c = nominal_attachment_cfg(cfg)
        _EPISODES["fom"] = grasp_episode(c)
        with tempfile.TemporaryDirectory(prefix="cable-full-space-") as directory:
            reduction = write_full_space_basis(c, directory)
            _EPISODES["rom"] = grasp_episode(c, reduction, read_modes(reduction.modes_path))
    return _EPISODES["fom"], _EPISODES["rom"]


def check_fixture(cfg, results):
    """Fixed base and planarity over the grasp episode (FOM and ROM); fixture placement
    semantics: once per run, identical repeats are no-ops, a move needs reset_fixture."""
    import Sofa.Simulation
    for name, run in zip(("FOM", "ROM"), stage_b_episodes(cfg)):
        dp, dtheta = run["base"].max(axis=0)
        results.append((f"{name}: fixed base over the run (<= 1e-9 m, 1e-9 rad)",
                        dp <= LIMITS["base_position"] and dtheta <= LIMITS["base_angle"],
                        f"max |dp| {dp:.1e} m, max |dtheta| {dtheta:.1e} rad over {len(run['base'])} steps"))
        results.append((f"{name}: planarity over the run (frames within 1e-8 m of the fixture plane)",
                        run["plane"].max() <= LIMITS["plane"], f"max distance {run['plane'].max():.1e} m"))
    c = nominal_attachment_cfg(cfg)
    root, cable, coupling = attached_scene(c, FIXTURE)
    for _ in range(4):
        Sofa.Simulation.animate(root, c["timestep_s"])
    placed = np.array(cable.base_mo.position.value, copy=True)
    coupling.on_fixture(list(FIXTURE))                   # same TF again: no-op
    repeat_ok = np.array_equal(placed, cable.base_mo.position.value)
    moved = list(FIXTURE)
    moved[0] += 1e-3
    try:
        coupling.on_fixture(moved)
        moved_refused = False
    except ValueError:
        moved_refused = np.array_equal(placed, cable.base_mo.position.value)
    coupling.reset_fixture(moved)
    Sofa.Simulation.animate(root, c["timestep_s"])
    reset_ok = np.abs(pose_delta(np.asarray(cable.base_mo.position.value[0], dtype=float),
                                 np.asarray(moved))).max() <= 1e-12
    results.append(("fixture placed once per run: identical repeat is a no-op, a move raises, "
                    "reset_fixture starts the new run", repeat_ok and moved_refused and reset_ok,
                    f"repeat unchanged {repeat_ok}, move refused {moved_refused}, reset placed {reset_ok}"))
    Sofa.Simulation.unload(root)


def check_grasp(cfg, results):
    """Equality attachment of s = L: residuals, identity, actual rows (FOM H_q, ROM H_a) vs FD,
    transmission (static balance), rejection of incompatible commands, release, FOM == ROM."""
    fom, rom = stage_b_episodes(cfg)
    for name, run in (("FOM", fom), ("ROM", rom)):
        dp = np.linalg.norm(run["end"][:, :2], axis=1).max()
        dyaw = np.abs(run["end"][:, 2]).max()
        results.append((f"{name}: latch only within 10 mm (12.8 mm refused, 5.4 mm latched); end "
                        "residual over the run (<= 1e-5 m, 1e-4 rad)",
                        run["latch_rejected_far"] and run["latched"] and dp <= LIMITS["end_position"]
                        and dyaw <= LIMITS["end_yaw"],
                        f"max |dp| {dp:.1e} m, max |dyaw| {dyaw:.1e} rad over {run['steps']} steps"))
        results.append((f"{name}: same end section and bitwise-identical offset until release",
                        run["identity"], f"identity held: {run['identity']}"))
        for hold, (error, sigma, nnz, rom_error) in zip(("hold 1", "hold 2"), run["rows"]):
            ok = error <= LIMITS["rows_fd"] and sigma[-1] > 1e-10 * sigma[0] and len(sigma) == 3
            detail = (f"scaled max|H - dC/dx_fd| / max|H| = {error:.1e}, sigma {sigma[0]:.3g} .. "
                      f"{sigma[-1]:.3g} (ratio {sigma[-1] / sigma[0]:.2e}), nnz {nnz}")
            label = "H_q (strain MO)" if rom_error is None else "H_a (modal MO)"
            if rom_error is not None:
                ok &= rom_error <= LIMITS["rows_rom"]
                detail += f"; max|H_a - H_q Phi| / max|H_a| = {rom_error:.1e}"
            results.append((f"{name} {hold}: actual {label} == FD dC/dx (<= 1e-6), row rank 3", ok, detail))
        results.append((f"{name}: plane-incompatible command (20 mm lift) refused, target untouched",
                        run["rejected_lift"], f"refused {run['rejected_lift']}"))
        results.append((f"{name}: release keeps positions/velocities bitwise, no rows or force after",
                        run["release_untouched"] and run["free_lambda"] == 0.0 and run["free_nnz"] == 0
                        and not run["active_after"],
                        f"state untouched {run['release_untouched']}, max |lambda| after "
                        f"{run['free_lambda']:.1e}, max nnz after {run['free_nnz']}"))
    for hold, (moment, force, m_scale, f_scale) in zip(("hold 1", "hold 2"), fom["balance"]):
        results.append((f"FOM {hold}: transmission W (q - q0) == H_q^T (lambda / h), moments and "
                        "forces separately (rel <= 1e-3)",
                        moment <= LIMITS["balance"] and force <= LIMITS["balance"],
                        f"moment rows rel {moment:.1e} (max {m_scale:.3e} N.m), force rows rel "
                        f"{force:.1e} (max {f_scale:.3e} N)"))
    frames = np.abs(fom["frames"] - rom["frames"]).max()
    reaction = [np.abs(fom["reactions"][:, rows] - rom["reactions"][:, rows]).max()
                / np.abs(fom["reactions"][:, rows]).max() for rows in ([0, 1], [2])]
    results.append(("full-space ROM == FOM over the episode: frames (<= 1e-8 m), reactions "
                    "(<= 1e-6 rel, force and moment)",
                    frames <= LIMITS["rom_frames"] and max(reaction) <= LIMITS["rom_reaction"],
                    f"max |frames| diff {frames:.1e} m, force rel {reaction[0]:.1e}, moment rel "
                    f"{reaction[1]:.1e} (max |F| {np.abs(fom['reactions'][:, :2]).max():.3f} N, max "
                    f"|Mz| {np.abs(fom['reactions'][:, 2]).max():.2e} N.m)"))


def check_checkpoint(cfg, results):
    """Rollout checkpoint: save mid-ramp, advance N steps, restore, replay the same N steps."""
    import Sofa.Simulation
    c = nominal_attachment_cfg(cfg)
    h = c["timestep_s"]
    root, cable, coupling = attached_scene(c, FIXTURE)
    command = PlanarCommand(FIXTURE, GRIPPER_LATCH, h)
    coupling.request_attach()
    coupling.update_grasp(*command.at(0, 0.0))
    command.move(-0.04, 0.05, 0.3)
    frames_node = cable.frames_mo.getContext()

    def advance(k, n):
        history = []
        for k in range(k, k + n):
            coupling.update_grasp(*command.at(k + 1, (k + 1) * h))
            Sofa.Simulation.animate(root, h)
            history.append((np.array(cable.frames_mo.position.value, dtype=float),
                            np.array(cable.frames_mo.velocity.value, dtype=float),
                            cable.attachment_reaction(), float(root.time.value)))
        return k + 1, history

    k, _ = advance(0, command.steps // 3)
    saved = {"k": k, "coupling": coupling.checkpoint()}
    n = 200
    k_a, first = advance(saved["k"], n)
    after_a = (k_a, float(root.time.value), float(frames_node.time.value),
               int(cable.attachment.index.value), coupling.offset)
    coupling.restore(saved["coupling"])
    restored_time = float(root.time.value) == saved["coupling"]["time"] == float(frames_node.time.value)
    k_b, second = advance(saved["k"], n)
    after_b = (k_b, float(root.time.value), float(frames_node.time.value),
               int(cable.attachment.index.value), coupling.offset)
    position = max(np.abs(a[0] - b[0]).max() for a, b in zip(first, second))
    velocity = max(np.abs(a[1] - b[1]).max() for a, b in zip(first, second))
    reaction = max(np.abs(a[2] - b[2]).max() for a, b in zip(first, second))
    times = all(a[3] == b[3] for a, b in zip(first, second))
    moving = np.linalg.norm(first[-1][0][-1, :3] - first[0][0][-1, :3])
    results.append((f"checkpoint mid-ramp -> {n} steps -> restore -> replay (<= 1e-12 m, 1e-12 N; "
                    "time, command index, attachment index and offset exact)",
                    position <= LIMITS["replay_position"] and reaction <= LIMITS["replay_reaction"]
                    and times and restored_time and after_a == after_b,
                    f"max |dx| {position:.1e} m, |dv| {velocity:.1e}, |dF| {reaction:.1e}; times equal "
                    f"{times}, restored time (root and child) {restored_time}, end state (k, t, index, "
                    f"offset) equal {after_a == after_b}; the end moved {1e3 * moving:.2f} mm during "
                    f"the window"))
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


CHECKS = {"traction": check_traction, "bending": check_bending, "kinematics": check_kinematics,
          "transpose": check_transpose, "restore": check_restore, "fixture": check_fixture,
          "grasp": check_grasp, "checkpoint": check_checkpoint, "energy": check_energy,
          "convergence": check_convergence}


def main(argv=None):
    import argparse
    from cable_identification import cosserat_model as cm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--checks", nargs="+", choices=list(CHECKS), default=list(CHECKS))
    args = parser.parse_args(argv)
    cfg = cm.load_config(args.config)
    results = []
    for name in args.checks:
        CHECKS[name](cfg, results)
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

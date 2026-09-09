"""Every term of  M(q) q'' + C(q,q') q' + f_int(q; EA,EI,GJ) = f_g + J_g(q)^T lambda
as SOFA assembles it for the Cosserat truth cable.

Builds the cable with ``expose_matrices=True`` (MatrixLinearSystem observers on the
solver node), attaches the grasp, drags the tip and reads the LAST step, whose forces,
matrices and Jacobian SOFA evaluates at the state q, q' of its start:

  q, q'         MechanicalObject position/velocity (rigid base 6 dofs + 16 x 3 strains)
  q''           (q'+ - q')/h, q'+ - q' = SparseLDLSolver solution (MatrixLinearSystem.x)
  M(q)          MatrixLinearSystem(assembleMass) / (1 + h rM): J^T diag(m) J of UniformMass
  C(q,q') q'    EulerImplicitSolver Rayleigh damping (rM M - rK K) q'+ - rK_ff K_int dq'
                (implicit; SOFA computes no Coriolis/centrifugal term for mapped masses)
  f_int         BeamHookeLawForceField: -K_int (q - q0), K_int from
                MatrixLinearSystem(assembleStiffness, applyMappedComponents=0)
  f_g           UniformMass x root.gravity (0: table model) + base clamp reaction
  lambda        FramesMO.force[tip]: RestShapeSpringsForceField wrench (F, tau)
  J_g(q)        DiscreteCosseratMapping.applyJ probed with unit velocities (the J SOFA uses)
  J_g^T lambda  DiscreteCosseratMapping.applyJT (strain/base force minus Hooke and clamp)

The derivative of the mapping's apply() (central differences) is computed as well: the
pinned Cosserat release's applyJ is NOT that derivative (it rotates the distal rod about
the section START node, lever l (L - s_i) instead of l (L - s_i - l/2) on a straight
rod), so J_g^T lambda, J^T M J and J^T K_g J are all built with a lumped-node Jacobian.
The dump measures and asserts that property instead of hiding it.

Every matrix is also written by SOFA's own GlobalSystemMatrixExporter (SofaMatrix) and
compared with the in-process read; the identities SOFA guarantees are asserted.

With ``--rom`` the same thing is done for the POD-Galerkin reduced model (the promoted
strain basis Phi of artifacts/cable_mor, or ``--modes/--modes-metadata`` for another
candidate): the strains are ``kappa = kappa_0 + Phi a`` through ModelOrderReductionMapping,
the unknowns are the base (6) and the modal coordinates a (r), and SOFA assembles
``M_r = J_r^T diag(m) J_r``, ``K_r = Phi^T K Phi`` exactly as the full model does. The
reduced matrices SOFA writes are checked against ``Phi^T (.) Phi`` of the full model at
the same physical configuration and against the POD metadata (orthonormality, energy).

Usage:  cable_dynamics_dump [--config cable_truth.yaml] [--output DIR]
        [--drag-steps 100] [--tip-offset 0.04 0.06] [--rom [--modes F --modes-metadata F]]
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np

from cable_identification.coupling import _q_conj, _q_mul


NB = 6  # Rigid3d base: 3 translations + 3 rotations = first block of every vector/matrix


def default_config():
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory("cable_identification"),
                        "config", "cable_truth.yaml")


def rotvec(q):
    """World-frame rotation vector of a unit quaternion (x, y, z, w)."""
    q = np.asarray(q, dtype=float)
    if q[3] < 0:
        q = -q
    n = np.linalg.norm(q[:3])
    return np.zeros(3) if n < 1e-300 else q[:3] / n * 2.0 * np.arctan2(n, q[3])


def quat(rv):
    a = np.linalg.norm(rv)
    return np.array([0., 0., 0., 1.]) if a < 1e-300 else np.r_[rv / a * np.sin(a / 2), np.cos(a / 2)]


def pose_delta(pose, ref):
    """(dp, dtheta_world) of pose w.r.t. ref, both [x y z qx qy qz qw]."""
    return np.r_[pose[:3] - ref[:3], rotvec(_q_mul(pose[3:7], _q_conj(ref[3:7])))]


def tip_jacobian_fd(cable, base7, strains, eps=1e-6):
    """6 x (6 + 3 ns) derivative of SOFA's apply() at the tip frame, central differences.

    Base rotation columns perturb the base quaternion by a world-frame rotation
    (SOFA Rigid3d convention: angular velocity/torque in world coordinates).
    """
    base7, strains = np.asarray(base7, dtype=float), np.asarray(strains, dtype=float)

    def tip_at(b7, s):
        with cable.base_mo.position.writeable() as p:
            p[0] = b7
        with cable.strain_mo.position.writeable() as x:
            x[:] = np.reshape(s, strains.shape)
        cable.refresh_mapping()
        return np.array(cable.tip_pose())

    def column(plus, minus):
        return pose_delta(tip_at(*plus), tip_at(*minus)) / (2 * eps)

    cols = []
    for i in range(3):
        d = np.zeros(3); d[i] = eps
        cols.append(column((np.r_[base7[:3] + d, base7[3:]], strains),
                           (np.r_[base7[:3] - d, base7[3:]], strains)))
    for i in range(3):
        rv = np.zeros(3); rv[i] = eps
        cols.append(column((np.r_[base7[:3], _q_mul(quat(rv), base7[3:])], strains),
                           (np.r_[base7[:3], _q_mul(quat(-rv), base7[3:])], strains)))
    flat = strains.ravel()
    for k in range(flat.size):
        d = np.zeros(flat.size); d[k] = eps
        cols.append(column((base7, flat + d), (base7, flat - d)))
    tip_at(base7, strains)
    return np.array(cols).T


def tip_jacobian_sofa(cable):
    """6 x (6 + 3 ns) Jacobian SOFA uses: applyJ (run by mapping.init) on unit velocities."""
    saved = (cable.base_mo.velocity.value.copy(), cable.strain_mo.velocity.value.copy())
    n_strain = saved[1].size
    cols = []
    for k in range(NB + n_strain):
        with cable.base_mo.velocity.writeable() as vb, cable.strain_mo.velocity.writeable() as vs:
            vb[:] = 0.0
            vs[:] = 0.0
            if k < NB:
                vb[0][k] = 1.0
            else:
                vs.reshape(-1)[k - NB] = 1.0
        cable.refresh_mapping()
        cols.append(np.array(cable.frames_mo.velocity.value[-1]))
    with cable.base_mo.velocity.writeable() as vb, cable.strain_mo.velocity.writeable() as vs:
        vb[:] = saved[0]
        vs[:] = saved[1]
    cable.refresh_mapping()
    return np.array(cols).T


def read_export(path):
    """GlobalSystemMatrixExporter txt: '[' newline, one '[ a b c ]' row per line, ' ]'."""
    rows = [line.strip(" []\n") for line in Path(path).read_text().splitlines()]
    return np.array([[float(x) for x in row.split()] for row in rows if row])


def add_exporters(cable, export_dir):
    export_dir.mkdir(parents=True, exist_ok=True)
    for stale in glob.glob(str(export_dir / "*.txt")):
        os.remove(stale)
    return [cable.solver_node.addObject(
        "GlobalSystemMatrixExporter", name=f"export{key}", linearSystem=system.getLinkPath(),
        filename=str(export_dir / key), format="txt", precision=17,
        exportEveryNumberOfSteps=1, enable=False)
        for key, system in cable.linear_systems.items()]


def read_exports(cable, export_dir):
    return {key: read_export(sorted(glob.glob(str(export_dir / f"{key}0*.txt")))[-1])
            for key in cable.linear_systems}


def drive(cable, coupling, cfg, drag_steps, tip_offset, exporters):
    """Attach and drag; returns (state before the last step, n_steps) after that step."""
    import Sofa.Simulation
    root = cable.solver_node.getRoot()
    coupling.request_attach()
    h = float(cfg["timestep_s"])
    tip0 = cable.tip_pose()
    goal = [tip0[0] - tip_offset[0], tip0[1] + tip_offset[1], tip0[2], 0.0, 0.0, 0.0, 1.0]
    n_steps = round(float(cfg["attach_ramp_s"]) / h) + drag_steps
    for k in range(n_steps - 1):
        coupling.update_grasp(goal, (k + 1) * h)
        Sofa.Simulation.animate(root, h)
    before = cable.save_state()
    before["base_velocity"] = np.asarray(cable.base_mo.velocity.value).ravel().copy()
    before["strain_position"] = np.asarray(cable.strain_mo.position.value).ravel().copy()
    before["tip"] = np.array(cable.tip_pose())
    for exporter in exporters:
        exporter.enable.value = True
    coupling.update_grasp(goal, n_steps * h)
    Sofa.Simulation.animate(root, h)
    return before, n_steps


def collect(cfg, drag_steps, tip_offset, export_dir):
    import Sofa.Core
    import Sofa.Simulation
    from cable_identification import cosserat_model as cm
    from cable_identification.coupling import GraspCoupling

    root = Sofa.Core.Node("cable_dynamics")
    cm.prepare_root(root, cfg, extra_plugins=["SofaMatrix"])
    cable = cm.build_cable(root, cfg, expose_matrices=True)
    exporters = add_exporters(cable, export_dir)
    coupling = GraspCoupling(cable, attach_mode="explicit")
    Sofa.Simulation.init(root)
    coupling.on_fixture([0.0] * 6 + [1.0])
    # straight rod: where the lumped-node Jacobian of the plugin has a closed form
    straight = {"J_sofa": tip_jacobian_sofa(cable),
                "J_fd": tip_jacobian_fd(cable, cable.base_mo.position.value[0],
                                        cable.strain_mo.position.value)}
    before, n_steps = drive(cable, coupling, cfg, drag_steps, tip_offset, exporters)

    # State at the START of the last step: that is where SOFA evaluates f, M, K and J.
    q_base = np.asarray(before["base"]).ravel()
    q_strain = np.asarray(before["strain"])
    v_old = np.r_[before["base_velocity"], np.asarray(before["strain_velocity"]).ravel()]
    tip = before["tip"]

    s = cable.system_matrices()
    fa = s["factors"]
    f_total = np.r_[np.asarray(cable.base_mo.force.value).ravel(),
                    np.asarray(cable.strain_mo.force.value).ravel()]
    v_new = np.r_[np.asarray(cable.base_mo.velocity.value).ravel(),
                  np.asarray(cable.strain_mo.velocity.value).ravel()]
    dv = np.asarray(s["x"], dtype=float)
    frames_force = np.asarray(cable.frames_mo.force.value)
    q0_strain = np.asarray(cable.strain_mo.rest_position.value)
    base_rest = np.asarray(cable.base_mo.rest_position.value).ravel()
    target = np.asarray(cable.grasp_target_mo.position.value).ravel()
    exported = read_exports(cable, export_dir)
    base_spring = cable.solver_node.rigidBase.baseSpring

    n = f_total.size
    f_hooke = np.zeros(n)
    f_hooke[NB:] = s["K_int"][NB:, NB:] @ (q_strain.ravel() - q0_strain.ravel())
    lam = frames_force[-1].copy()
    cable.restore_state(before)
    cable.refresh_mapping()
    J = tip_jacobian_sofa(cable)
    J_fd = tip_jacobian_fd(cable, q_base, q_strain)
    # base rows: SOFA's force minus the projected gripper wrench is the clamp reaction
    f_clamp = np.zeros(n)
    f_clamp[:NB] = f_total[:NB] - (J.T @ lam)[:NB]
    jt_lambda = f_total - f_hooke - f_clamp
    gravity = np.asarray(root.gravity.value, dtype=float)
    section_start = np.r_[0.0, np.cumsum(np.asarray(cable.force_field.length.value))[:-1]]

    M, K, K_int = s["M"], s["K"], s["K_int"]
    q_dd = dv / fa["h"]
    c_qdot = (fa["rM"] * M - fa["rK"] * K) @ v_new - fa["rK_ff"] * K_int @ dv
    out = {
        "h": np.float64(fa["h"]), "n_steps": np.int64(n_steps),
        "q_base": q_base, "q_strain": q_strain, "q0_strain": q0_strain, "base_rest": base_rest,
        "v": v_old, "v_plus": v_new, "dv": dv, "q_dd": q_dd,
        "M": M, "K": K, "K_int": K_int, "K_clamp": s["K_clamp"], "K_grasp": s["K_grasp"],
        "A": s["A"].toarray(), "b": np.asarray(s["b"], dtype=float),
        "M_term": s["M_term"], "B_term": s["B_term"], "K_term": s["K_term"],
        "K_direct_term": s["K_direct_term"],
        "M_qdd": M @ q_dd, "C_qdot": c_qdot,
        "C_qdot_increment_part": -fa["rK_ff"] * K_int @ dv,
        "f_int": -f_hooke, "f_g": np.zeros(n), "gravity": gravity, "f_clamp": f_clamp,
        "clamp_delta": pose_delta(q_base, base_rest),
        "lambda": lam, "J_g": J, "J_g_fd": J_fd, "Jt_lambda": jt_lambda,
        "Jt_lambda_J": J.T @ lam, "Jt_lambda_fd": J_fd.T @ lam,
        "straight_J_sofa": straight["J_sofa"], "straight_J_fd": straight["J_fd"],
        "section_start": section_start, "section_length": np.asarray(cable.force_field.length.value),
        "length_m": np.float64(cfg["length_m"]),
        "f_total": f_total, "frames_force": frames_force,
        "linearisation": fa["h"] * K @ v_new,
        "tip": tip, "target": target, "frames": np.asarray(cable.frame_poses()),
        "grasp_stiffness": np.array([float(cable.grasp_spring.stiffness.value[0]),
                                     float(cable.grasp_spring.angularStiffness.value[0])]),
        "clamp_stiffness": np.array([float(base_spring.stiffness.value[0]),
                                     float(base_spring.angularStiffness.value[0])]),
        "hooke_section": np.array([float(cable.force_field.GI.value), float(cable.force_field.EI.value),
                                   float(cable.force_field.EI.value)])
        * np.asarray(cable.force_field.length.value)[:, None],
        "mass_kg": np.float64(cfg["mass_kg"]),
        "locked_dofs": np.flatnonzero(np.all(s["A"].toarray() == np.eye(n), axis=1)),
    }
    out.update({f"factor_{k}": np.float64(v) for k, v in fa.items()})
    out.update({f"export_{k}": v for k, v in exported.items()})
    # M q'' + C q' + f_int - f_g - f_clamp - J^T lambda - (f(q+) - f(q)) == 0 (SOFA identity)
    out["residual"] = (out["M_qdd"] + out["C_qdot"] + out["f_int"] - out["f_g"] - out["f_clamp"]
                       - out["Jt_lambda"] - out["linearisation"])
    Sofa.Simulation.unload(root)
    return out


def collect_rom(cfg, reduction, drag_steps, tip_offset, export_dir):
    """POD-Galerkin ROM: the same drag with kappa = kappa_0 + Phi a, matrices from SOFA."""
    import Sofa.Core
    import Sofa.Simulation
    from cable_identification import cosserat_model as cm
    from cable_identification.coupling import GraspCoupling
    from cable_identification.strain_basis import validate_reduction

    _, phi, metadata = validate_reduction(cfg, reduction)
    root = Sofa.Core.Node("cable_dynamics_rom")
    cm.prepare_root(root, cfg, extra_plugins=["SofaMatrix", "ModelOrderReduction"])
    cable = cm.build_cable(root, cfg, reduction=reduction, expose_matrices=True)
    exporters = add_exporters(cable, export_dir)
    coupling = GraspCoupling(cable, attach_mode="explicit")
    Sofa.Simulation.init(root)
    coupling.on_fixture([0.0] * 6 + [1.0])
    before, n_steps = drive(cable, coupling, cfg, drag_steps, tip_offset, exporters)

    s = cable.system_matrices()
    fa = s["factors"]
    r = phi.shape[1]
    a = np.asarray(before["modal"]).ravel()
    a_dot = np.asarray(before["modal_velocity"]).ravel()
    v_r_old = np.r_[before["base_velocity"], a_dot]
    v_r_new = np.r_[np.asarray(cable.base_mo.velocity.value).ravel(),
                    np.asarray(cable.modal_mo.velocity.value).ravel()]
    da = np.asarray(s["x"], dtype=float)
    f_r = np.r_[np.asarray(cable.base_mo.force.value).ravel(),
                np.asarray(cable.modal_mo.force.value).ravel()]
    kappa = before["strain_position"]  # SOFA evaluated f, M, K at the start of the step
    kappa0 = np.asarray(cable.strain_mo.rest_position.value).ravel()
    hooke = np.repeat(np.array([float(cable.force_field.GI.value), float(cable.force_field.EI.value),
                                float(cable.force_field.EI.value)])[None], phi.shape[0] // 3, 0) \
        * np.asarray(cable.force_field.length.value)[:, None]
    k_int_full = -np.diag(hooke.ravel())
    f_hooke_full = k_int_full @ (kappa - kappa0)
    M_r, A_r = s["M"], s["A"].toarray()
    # kappa = kappa0 + Phi a: the FOM at the ROM's configuration gives the Galerkin identities
    fom = full_at(cfg, np.asarray(before["base"]).ravel(), np.reshape(kappa0 + phi @ a, (-1, 3)),
                  cable.grasp_target_mo.position.value[0], a_dot, phi)
    proj = np.zeros((NB + r, NB + phi.shape[0]))
    proj[:NB, :NB] = np.eye(NB)
    proj[NB:, NB:] = phi.T
    # the mapped stiffness carries two factors: Hooke -h(h+rK+rK_ff) K_int, grasp -h(h+rK) K_g
    k_int_r_term = fa["kF_hooke"] * (phi.T @ k_int_full @ phi)
    k_grasp_r = (s["K_mapped_term"] - np.pad(k_int_r_term, ((NB, 0), (NB, 0)))) / fa["kF_springs"]
    K_r = phi.T @ k_int_full @ phi
    K_r_full = np.pad(K_r, ((NB, 0), (NB, 0))) + s["K_clamp"] + k_grasp_r
    sing = np.asarray(metadata["singular_values"], dtype=float)
    f_int_r = np.r_[np.zeros(NB), -phi.T @ f_hooke_full]
    f_clamp_r = np.r_[fom["f_clamp"][:NB], np.zeros(r)]
    out = {
        "h": np.float64(fa["h"]), "n_steps": np.int64(n_steps), "r": np.int64(r),
        "Phi": phi, "singular_values": sing,
        "energy_captured": np.float64((sing[:r] ** 2).sum() / (sing ** 2).sum()),
        "a": a, "a_dot": a_dot, "v_r": v_r_old, "v_r_plus": v_r_new, "da": da, "q_r_dd": da / fa["h"],
        "kappa": kappa, "kappa0": kappa0, "kappa_from_modes": kappa0 + phi @ a,
        "M_r": M_r, "K_r": K_r_full, "K_int_r": K_r, "K_grasp_r": k_grasp_r, "K_clamp_r": s["K_clamp"],
        "A_r": A_r, "b_r": np.asarray(s["b"], dtype=float),
        "M_r_term": s["M_term"], "B_r_term": s["B_term"], "K_r_term": s["K_term"],
        "K_r_direct_term": s["K_direct_term"], "K_r_mapped_term": s["K_mapped_term"],
        "M_r_qdd": M_r @ (da / fa["h"]),
        "C_r_qdot": (fa["rM"] * M_r - fa["rK"] * K_r_full) @ v_r_new
        - fa["rK_ff"] * np.pad(K_r, ((NB, 0), (NB, 0))) @ da,
        "f_int_r": f_int_r, "f_clamp_r": f_clamp_r, "f_r": f_r,
        # f_r = Hooke + clamp + J^T lambda, and f_int_r is MINUS the Hooke force
        "Jt_lambda_r": f_r + f_int_r - f_clamp_r,
        "linearisation_r": fa["h"] * K_r_full @ v_r_new,
        "f_strain_mapped": np.asarray(cable.strain_mo.force.value).ravel(),
        "lambda": np.asarray(cable.frames_mo.force.value)[-1].copy(),
        "tip": before["tip"], "target": np.asarray(cable.grasp_target_mo.position.value).ravel(),
        "frames": np.asarray(cable.frame_poses()),
        "projected_M": proj @ fom["M"] @ proj.T, "projected_K_term": proj @ fom["K_term"] @ proj.T,
        "fom_tip": fom["tip"],
        "locked_dofs": np.flatnonzero(np.all(A_r == np.eye(NB + r), axis=1)),
    }
    out["residual_r"] = (out["M_r_qdd"] + out["C_r_qdot"] + f_int_r - f_clamp_r - out["Jt_lambda_r"]
                         - out["linearisation_r"])
    out.update({f"factor_{k}": np.float64(v) for k, v in fa.items()})
    out.update({f"export_{k}": v for k, v in read_exports(cable, export_dir).items()})
    Sofa.Simulation.unload(root)
    return out


def full_at(cfg, base7, strains, target7, a_dot=None, phi=None):
    """FOM matrices at a prescribed configuration (one solve; velocities zero or Phi a_dot)."""
    import Sofa.Core
    import Sofa.Simulation
    from cable_identification import cosserat_model as cm
    from cable_identification.coupling import GraspCoupling

    root = Sofa.Core.Node("cable_dynamics_fom_at")
    cm.prepare_root(root, cfg)
    cable = cm.build_cable(root, cfg, expose_matrices=True)
    coupling = GraspCoupling(cable, attach_mode="explicit")
    Sofa.Simulation.init(root)
    coupling.on_fixture([0.0] * 6 + [1.0])
    coupling.request_attach()
    coupling.update_grasp(cable.tip_pose(), 0.0)
    coupling.ramp_s = 0.0
    with cable.strain_mo.position.writeable() as x:
        x[:] = strains
    if a_dot is not None:
        with cable.strain_mo.velocity.writeable() as v:
            v[:] = np.reshape(phi @ a_dot, strains.shape)
    cable.refresh_mapping()
    cable.set_grasp_pose(list(target7))
    coupling.update_grasp(list(target7), 1.0)
    tip = np.array(cable.tip_pose())
    Sofa.Simulation.animate(root, float(cfg["timestep_s"]))
    s = cable.system_matrices()
    A = s["A"].toarray()
    lam = np.asarray(cable.frames_mo.force.value)[-1]
    f_base = np.asarray(cable.base_mo.force.value).ravel()
    J = tip_jacobian_sofa(cable)
    out = {"M": s["M"], "K_term": s["K_term"], "A": A, "tip": tip,
           "f_clamp": f_base - (J.T @ lam)[:NB],
           "locked": np.flatnonzero(np.all(A == np.eye(A.shape[0]), axis=1))}
    Sofa.Simulation.unload(root)
    return out


def check(d):
    """The identities SOFA guarantees; if one fails the extraction no longer means this."""
    n = d["M"].shape[0]
    free = np.setdiff1d(np.arange(n), d["locked_dofs"])
    ix = np.ix_(free, free)
    close = lambda a, b, rtol, atol=0.0: np.allclose(a, b, rtol=rtol, atol=atol)  # noqa: E731
    # what the exporter wrote (17 significant digits) == what the binding read
    for key, term in (("A", "A"), ("M", "M_term"), ("K", "K_term"), ("K_direct", "K_direct_term")):
        assert close(d[f"export_{key}"], d[term], 1e-15, 1e-15 * np.abs(d[term]).max()), key
    # the solved system and its three observers
    assert close(d["A"][ix], (d["M_term"] + d["B_term"] + d["K_term"])[ix], 1e-12, 1e-9)
    assert np.linalg.norm(d["A"] @ d["dv"] - d["b"]) <= 1e-8 * np.linalg.norm(d["b"])
    assert close(d["v_plus"] - d["v"], d["dv"], 1e-9, 1e-12)
    assert not d["B_term"].any()
    # M: symmetric positive definite, cable mass on the base translation block
    assert close(d["M"], d["M"].T, 0, 1e-12) and np.linalg.eigvalsh(d["M"]).min() > 0
    assert close(d["M"][:3, :3].diagonal(), float(d["mass_kg"]), 1e-9)
    # K split: Hooke law per section, clamp on the base, grasp spring stiffness projected
    ns = d["q_strain"].shape[0]
    assert close(d["K_int"][NB:, NB:], -np.diag(d["hooke_section"].ravel()), 1e-12, 1e-300)
    assert not d["K_int"][:NB].any() and not d["K_clamp"][NB:].any()
    assert close(d["K_clamp"][:NB, :NB], -np.diag(np.repeat(d["clamp_stiffness"], 3)), 1e-12)
    assert abs(np.abs(d["K_grasp"]).max() - d["grasp_stiffness"][0]) <= 1e-6 * d["grasp_stiffness"][0]
    # lambda is the grasp spring wrench and the only force on the frames
    assert not d["frames_force"][:-1].any()
    k, k_a = d["grasp_stiffness"]
    assert close(d["lambda"][:3], k * (d["target"][:3] - d["tip"][:3]), 1e-9, 1e-12)
    assert close(d["lambda"][3:], -k_a * rotvec(_q_mul(d["tip"][3:7], _q_conj(d["target"][3:7]))),
                 1e-9, 1e-12)
    # applyJT is the transpose of applyJ: J^T lambda from the probed J == force - Hooke - clamp
    assert close(d["Jt_lambda_J"], d["Jt_lambda"], 1e-9, 1e-12 * np.abs(d["lambda"]).max())
    # clamp translation follows the spring law (rotation: SOFA's RestShapeSprings deadband)
    assert close(d["f_clamp"][:3], d["K_clamp"][:3, :3] @ d["clamp_delta"][:3], 1e-9, 1e-12)
    # SOFA's J vs the derivative of its apply(): rigid base and angular rows agree ...
    J, J_fd = d["J_g"], d["J_g_fd"]
    assert close(J[:3, :3], np.eye(3), 0, 1e-9) and close(J[3:], J_fd[3:], 1e-8, 1e-9)
    # ... the linear rows do not: on the straight rod the plugin uses lever l (L - s_i)
    ns = d["q_strain"].shape[0]
    bend_z = NB + 3 * np.arange(ns) + 2
    lever_sofa = d["section_length"] * (float(d["length_m"]) - d["section_start"])
    assert close(d["straight_J_sofa"][1, bend_z], lever_sofa, 1e-9)
    lever_true = d["section_length"] * (float(d["length_m"]) - d["section_start"] - d["section_length"] / 2)
    assert np.all(d["straight_J_fd"][1, bend_z] < lever_sofa) and close(d["straight_J_fd"][1, bend_z],
                                                                        lever_true, 5e-2)
    # the equation balances on the free dofs
    scale = max(np.linalg.norm(d["M_qdd"][free]), np.linalg.norm(d["f_total"][free]))
    assert np.linalg.norm(d["residual"][free]) <= 1e-10 * scale, np.linalg.norm(d["residual"][free])
    # planar table: torsion and bend_y of every section locked, bend_z free
    assert np.array_equal(d["locked_dofs"], np.array([NB + 3 * i + j for i in range(ns) for j in (0, 1)]))
    assert not d["gravity"].any() and not d["f_g"].any()
    assert d["M_qdd"].shape == d["C_qdot"].shape == d["f_int"].shape == (n,) and d["J_g"].shape == (6, n)


def check_rom(d):
    """POD-Galerkin identities of the reduced system SOFA assembled."""
    r, phi = int(d["r"]), d["Phi"]
    n = NB + r
    close = lambda a, b, rtol, atol=0.0: np.allclose(a, b, rtol=rtol, atol=atol)  # noqa: E731
    for key, term in (("A", "A_r"), ("M", "M_r_term"), ("K", "K_r_term"), ("K_direct", "K_r_direct_term")):
        assert close(d[f"export_{key}"], d[term], 1e-15, 1e-15 * np.abs(d[term]).max()), key
    # basis: orthonormal columns, planar rows zero, POD energy monotone
    assert close(phi.T @ phi, np.eye(r), 0, 1e-3) and not d["locked_dofs"].size
    assert np.all(np.diff(d["singular_values"]) <= 0) and 0 < d["energy_captured"] <= 1
    # the mapping: kappa = kappa0 + Phi a, exactly what the strain MO holds
    assert close(d["kappa"], d["kappa_from_modes"], 1e-9, 1e-12)
    # the solved reduced system
    assert close(d["A_r"], d["M_r_term"] + d["B_r_term"] + d["K_r_term"], 1e-12, 1e-9)
    assert np.linalg.norm(d["A_r"] @ d["da"] - d["b_r"]) <= 1e-8 * np.linalg.norm(d["b_r"])
    assert close(d["v_r_plus"] - d["v_r"], d["da"], 1e-9, 1e-12) and not d["B_r_term"].any()
    # Galerkin: SOFA's reduced mass/stiffness == Phi^T (FOM at the same configuration) Phi
    # (1e-8: Phi is read back from a 5-decimal text file; the FOM has a 1e8 clamp spring)
    assert close(d["M_r"], d["projected_M"], 1e-8, 1e-8 * np.abs(d["M_r"]).max())
    assert close(d["K_r_term"], d["projected_K_term"], 1e-8, 1e-8 * np.abs(d["K_r_term"]).max())
    assert close(d["fom_tip"][:3], d["tip"][:3], 0, 1e-6)
    assert close(d["M_r"], d["M_r"].T, 0, 1e-12) and np.linalg.eigvalsh(d["M_r"]).min() > 0
    # the reduced internal stiffness is Phi^T diag(GI l, EI l, EI l) Phi (negative definite)
    assert np.linalg.eigvalsh(d["K_int_r"]).max() < 0
    assert abs(np.abs(d["K_grasp_r"]).max()) <= 2000.0 * (1 + 1e-6)
    # the reduced equation balances
    scale = max(np.linalg.norm(d["M_r_qdd"]), np.linalg.norm(d["f_r"]))
    assert np.linalg.norm(d["residual_r"]) <= 1e-10 * scale, np.linalg.norm(d["residual_r"])
    assert d["M_r"].shape == (n, n) and d["f_int_r"].shape == (n,)


def _row(sym, source, x):
    x = np.asarray(x)
    return (f"  {sym:<14s} {source:<60s} {str(x.shape):<9s} "
            f"|.|max={np.abs(x).max():<11.4g} ||.||={np.linalg.norm(x):.4g}")


def write_report(d, path):
    h, ns = float(d["h"]), d["q_strain"].shape[0]
    fa = {k[7:]: float(v) for k, v in d.items() if k.startswith("factor_")}
    free = np.setdiff1d(np.arange(d["M"].shape[0]), d["locked_dofs"])
    lines = [
        "M(q) q'' + C(q,q') q' + f_int(q; EA, EI, GJ) = f_g + J_g(q)^T lambda   -- as SOFA assembles it",
        "",
        f"  dofs: 0..5 rigid base (tx ty tz rx ry rz), 6.. = {ns} sections x (torsion, bend_y, bend_z)",
        f"  planar table: PartialFixedProjectiveConstraint locks dofs {d['locked_dofs'].tolist()}",
        f"  step {int(d['n_steps'])}, h = {h}, rM = {fa['rM']}, rK = {fa['rK']}, "
        f"Hooke rayleighStiffness rK_ff = {fa['rK_ff']}",
        "",
        "  term           SOFA source                                                  shape     magnitude",
        _row("q (base)", "RigidBaseMO.position [x y z qx qy qz qw]", d["q_base"]),
        _row("q (strain)", "cosseratCoordinateMO.position [1/m], rest q0 = 0", d["q_strain"]),
        _row("q'", "MechanicalObject.velocity at the start of the step", d["v"]),
        _row("q''", "(q'+ - q')/h, q'+ - q' = SparseLDLSolver solution", d["q_dd"]),
        _row("M(q)", "MatrixLinearSystem(assembleMass)/(1+h rM) = J^T diag(m) J", d["M"]),
        _row("M q''", "", d["M_qdd"]),
        _row("C(q,q') q'", "EulerImplicit Rayleigh (rM M - rK K) q'+ - rK_ff K_int dq'", d["C_qdot"]),
        _row("  of which", "Hooke's own rayleighStiffness, matrix side: -rK_ff K_int dq'",
             d["C_qdot_increment_part"]),
        _row("f_int", "BeamHookeLawForceField: -K_int (q - q0)", d["f_int"]),
        _row("K_int", "MatrixLinearSystem(assembleStiffness, applyMapped=0), strains", d["K_int"]),
        _row("f_g", "UniformMass x root.gravity, gravity = " + str(d["gravity"].tolist()), d["f_g"]),
        _row("f_clamp", "baseSpring (RestShapeSprings 1e8): base force - J_g^T lambda", d["f_clamp"]),
        _row("lambda", "FramesMO.force[tip] = graspSpring wrench (Fx Fy Fz tx ty tz)", d["lambda"]),
        _row("J_g(q)", "DiscreteCosseratMapping.applyJ probed with unit velocities", d["J_g"]),
        _row("J_g^T lambda", "DiscreteCosseratMapping.applyJT (force - Hooke - clamp)", d["Jt_lambda"]),
        _row("J_g^T lambda", "probed J_g ^T lambda (== applyJT: transpose consistent)", d["Jt_lambda_J"]),
        _row("d apply/dq", "derivative of the mapping's apply(), central differences", d["J_g_fd"]),
        _row("(d apply/dq)^T lambda", "what a consistent Jacobian would project", d["Jt_lambda_fd"]),
        _row("J^T K_g J", "grasp spring stiffness projected on q (part of K)", d["K_grasp"]),
        _row("f(q+) - f(q)", "SOFA linearisation h K q'+ (implicit Euler)", d["linearisation"]),
        _row("residual", "M q'' + C q' + f_int - f_g - f_clamp - J^T lambda - (f(q+)-f(q))",
             d["residual"][free]),
        "",
        "  SOFA solves  A dq' = b  with  A = (1 + h rM) M - h B - h (h + rK [+ rK_ff]) K,",
        "  b = h (f + ((h + rK) K - rM M) q')   (EulerImplicitSolver.cpp v25.12; K = df/dq < 0 for springs).",
        "  Rewritten:  M q'' + (rM M - rK K) q'+ - rK_ff K_int dq' = f(q) + h K q'+  ~  f(q+),",
        "  i.e. the equation above at the new configuration. SOFA has no Coriolis/centrifugal",
        "  term: the frames' mass is projected J^T M J every step and d(J^T M J)/dt is dropped.",
        "",
        f"  lambda check: k (target - tip) = {(d['grasp_stiffness'][0] * (d['target'][:3] - d['tip'][:3])).tolist()}",
        f"  tip {np.array2string(d['tip'], precision=4)}  target {np.array2string(d['target'], precision=4)}",
        f"  clamp: base displacement {np.array2string(d['clamp_delta'], precision=3)} -> translation force"
        " = k dx; rotation torque 0 (RestShapeSpringsForceField ignores |dtheta| < ~1e-7 rad)",
        "",
        "  Jacobian of the pinned Cosserat release (DiscreteCosseratMapping):",
        "    applyJT == applyJ^T (checked), but applyJ is not the derivative of apply(). Straight rod,",
        "    tip dy per unit bend_z rate of section i:",
        "      SOFA applyJ      : " + np.array2string(d["straight_J_sofa"][1, NB + 2::3], precision=5),
        "      d apply/dq (FD)  : " + np.array2string(d["straight_J_fd"][1, NB + 2::3], precision=5),
        "      l (L - s_i)      : " + np.array2string(
            d["section_length"] * (float(d["length_m"]) - d["section_start"]), precision=5),
        "    i.e. the plugin rotates the distal rod about the section START node (lever l (L - s_i))",
        "    instead of distributing the curvature over the section (lever ~ l (L - s_i - l/2)).",
        "    At this bent configuration, bend_z rows of J_g^T lambda, SOFA / consistent:",
        "      " + np.array2string(d["Jt_lambda"][NB + 2::3] / d["Jt_lambda_fd"][NB + 2::3], precision=3),
        "    M = J^T diag(m) J, J^T K_g J and J_g^T lambda all use this J; the equation is exact for",
        "    SOFA's own J (residual below) but not the virtual work of the frames' motion.",
        "",
        "  SOFA-written matrices (GlobalSystemMatrixExporter, SofaMatrix): sofa_export/{A,M,K,K_direct}*.txt",
        "",
    ]
    np.set_printoptions(linewidth=250, precision=6, suppress=False, threshold=1_000_000)
    for name in ("q_base", "q_strain", "v", "q_dd", "M", "M_qdd", "C_qdot", "f_int", "K_int", "f_g",
                 "f_clamp", "lambda", "J_g", "J_g_fd", "Jt_lambda", "Jt_lambda_fd", "K_grasp", "K", "A",
                 "b", "dv", "residual"):
        lines += [f"=== {name} ===", np.array2string(d[name]), ""]
    Path(path).write_text("\n".join(lines))


def _heat(ax, X, title):
    from matplotlib.colors import SymLogNorm
    nz = np.abs(X[np.abs(X) > 1e-15])
    vmax = nz.max() if nz.size else 1.0
    # log colours over 8 decades: below that it is assembly round-off, not structure
    im = ax.imshow(X, cmap="RdBu_r", aspect="auto",
                   norm=SymLogNorm(linthresh=vmax * 1e-8, vmin=-vmax, vmax=vmax))
    ax.set_title(f"{title}  {X.shape}  nnz={np.count_nonzero(np.abs(X) > 1e-15)}", fontsize=9)
    if X.shape[0] == X.shape[1]:
        ax.axhline(NB - 0.5, color="k", lw=0.5)
    ax.axvline(NB - 0.5, color="k", lw=0.5)
    ax.figure.colorbar(im, ax=ax, fraction=0.046)


def _bars(ax, x, title, locked):
    x = np.asarray(x)
    colors = ["0.7" if i in locked else ("tab:orange" if i < NB else "tab:blue") for i in range(x.size)]
    ax.bar(range(x.size), x, color=colors)
    ax.axvline(NB - 0.5, color="k", lw=0.5)
    ax.set_title(f"{title}   ||.||={np.linalg.norm(x):.3g}", fontsize=9)
    ax.set_xlabel("dof (orange: base, grey: locked)", fontsize=7)


def write_figure(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    locked = set(d["locked_dofs"].tolist())
    fig, ax = plt.subplots(3, 4, figsize=(26, 15))
    _heat(ax[0, 0], d["M"], "M(q) [kg, kg m, kg m^2]")
    _heat(ax[0, 1], d["K_int"], "K_int = d f_int/dq (Hooke, diag GI l, EI l, EI l)")
    _heat(ax[0, 2], d["K_grasp"], "J_g^T K_grasp J_g (gripper spring on q)")
    _heat(ax[0, 3], d["J_g"], "J_g(q) = applyJ, tip frame (rows: dx dy dz dthx dthy dthz)")
    _bars(ax[1, 0], d["M_qdd"], "M q''", locked)
    _bars(ax[1, 1], d["C_qdot"], "C(q,q') q'  (Rayleigh)", locked)
    _bars(ax[1, 2], d["f_int"], "f_int = -K_int (q - q0)", locked)
    _bars(ax[1, 3], d["Jt_lambda"], "J_g^T lambda  (+ f_clamp on base dofs, f_g = 0)", locked)
    a = ax[2, 0]
    a.bar(range(6), d["lambda"], color=["tab:red"] * 3 + ["tab:purple"] * 3)
    a.set_xticks(range(6)); a.set_xticklabels(["Fx", "Fy", "Fz", "tx", "ty", "tz"])
    a.set_title("lambda = gripper wrench on the tip [N, N m]", fontsize=9)
    a = ax[2, 1]
    sec = np.arange(d["q_strain"].shape[0])
    for j, lab in enumerate(("torsion", "bend_y", "bend_z")):
        a.plot(sec, d["q_strain"][:, j], "o-", label=f"q {lab} [1/m]")
    a2 = a.twinx()
    a2.plot(sec, d["v"][NB:].reshape(-1, 3)[:, 2], "x--", color="tab:green", label="q' bend_z [1/(m s)]")
    a2.plot(sec, d["q_dd"][NB:].reshape(-1, 3)[:, 2], "s:", color="tab:red", label="q'' bend_z [1/(m s^2)]")
    a.set_title("q, q', q'' per section", fontsize=9); a.set_xlabel("section")
    a.legend(loc="upper left", fontsize=7); a2.legend(loc="upper right", fontsize=7)
    a = ax[2, 2]
    sec = np.arange(d["q_strain"].shape[0])
    a.plot(sec, d["Jt_lambda"][NB + 2::3], "o-", label="SOFA applyJT (lumped-node J)")
    a.plot(sec, d["Jt_lambda_fd"][NB + 2::3], "s--", label="(d apply/dq)^T lambda")
    a.plot(sec, d["f_int"][NB + 2::3], "x:", label="f_int")
    a.set_title("bend_z rows: J_g^T lambda, SOFA vs consistent Jacobian [N m]", fontsize=9)
    a.set_xlabel("section"); a.legend(fontsize=7)
    a = ax[2, 3]
    fr = d["frames"]
    a.plot(fr[:, 0], fr[:, 1], "o-", ms=3, label="frames (after the step)")
    a.plot(*d["target"][:2], "r*", ms=12, label="grasp target")
    F = d["lambda"][:3]
    a.annotate("", xy=(d["tip"][0] + 0.1 * F[0], d["tip"][1] + 0.1 * F[1]), xytext=tuple(d["tip"][:2]),
               arrowprops=dict(color="tab:red", width=1.5))
    a.set_aspect("equal"); a.set_title("configuration (xy) and lambda force at the tip", fontsize=9)
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def write_rom_report(d, path):
    r, phi = int(d["r"]), d["Phi"]
    fa = {k[7:]: float(v) for k, v in d.items() if k.startswith("factor_")}
    sing = d["singular_values"]
    lines = [
        "POD-Galerkin ROM:  M_r(a) a'' + C_r a' + f_int_r(a) = f_g_r + J_r^T lambda   -- as SOFA assembles it",
        "",
        f"  kappa = kappa_0 + Phi a,  Phi {phi.shape} (48 planar strains x r = {r} POD modes),"
        f" ModelOrderReductionMapping",
        f"  unknowns: 0..5 rigid base, 6..{NB + r - 1} modal coordinates a  (no locked dofs: the planar"
        " rows of Phi are zero)",
        f"  POD singular values (first {min(8, sing.size)}): {np.array2string(sing[:8], precision=4)}",
        f"  energy captured by r = {r}: {float(d['energy_captured']):.6f}",
        f"  step {int(d['n_steps'])}, h = {fa['h']}, rM = {fa['rM']}, rK = {fa['rK']}, rK_ff = {fa['rK_ff']}",
        "",
        "  term           SOFA source                                                  shape     magnitude",
        _row("a", "modalCoordinateMO.position (start of the step)", d["a"]),
        _row("a'", "modalCoordinateMO.velocity", d["a_dot"]),
        _row("a''", "(a'+ - a')/h, solution of the reduced system", d["q_r_dd"][NB:]),
        _row("kappa", "cosseratCoordinateMO.position == kappa_0 + Phi a", d["kappa"]),
        _row("M_r", "MatrixLinearSystem(assembleMass)/(1+h rM) = Phi^T J^T m J Phi", d["M_r"]),
        _row("Phi^T M Phi", "FOM mass at the same configuration, projected (== M_r)", d["projected_M"]),
        _row("K_int_r", "Phi^T K_int Phi  (Hooke law in modal coordinates)", d["K_int_r"]),
        _row("K_grasp_r", "grasp spring through Cosserat mapping and Phi", d["K_grasp_r"]),
        _row("K_clamp_r", "base clamp (unchanged, base dofs are not reduced)", d["K_clamp_r"]),
        _row("M_r a''", "", d["M_r_qdd"]),
        _row("C_r a'", "Rayleigh (rM M_r - rK K_r) v+ - rK_ff K_int_r da", d["C_r_qdot"]),
        _row("f_int_r", "-Phi^T K_int (kappa - kappa_0) = modalCoordinateMO.force part", d["f_int_r"]),
        _row("J_r^T lambda", "Phi^T J_g^T lambda (mapped grasp wrench in modal coords)", d["Jt_lambda_r"]),
        _row("lambda", "FramesMO.force[tip] = graspSpring wrench", d["lambda"]),
        _row("A_r", "reduced implicit system, SparseLDLSolver", d["A_r"]),
        _row("b_r", "", d["b_r"]),
        _row("residual_r", "M_r a'' + C_r a' + f_int_r - f_clamp - J_r^T lambda - h K_r v+", d["residual_r"]),
        "",
        "  Galerkin identity checked: SOFA's assembled M_r and K_r_term equal Phi^T (FOM) Phi with the FOM",
        "  placed at kappa_0 + Phi a (same base, same gripper target): the reduction is intrusive, the",
        "  reduced operators are projections of the full ones, not fits.",
        f"  tip ROM {np.array2string(d['tip'][:3], precision=5)}  tip FOM at Phi a {np.array2string(d['fom_tip'][:3], precision=5)}",
        "",
        "  SOFA-written matrices: sofa_export_rom/{A,M,K,K_direct}*.txt",
        "",
    ]
    np.set_printoptions(linewidth=250, precision=6, suppress=False, threshold=1_000_000)
    for name in ("Phi", "singular_values", "a", "a_dot", "kappa", "M_r", "K_int_r", "K_grasp_r", "K_r",
                 "A_r", "b_r", "da", "f_int_r", "Jt_lambda_r", "residual_r"):
        lines += [f"=== {name} ===", np.array2string(d[name]), ""]
    Path(path).write_text("\n".join(lines))


def write_rom_figure(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = int(d["r"])
    fig, ax = plt.subplots(2, 4, figsize=(26, 10))
    _heat(ax[0, 0], d["Phi"], "Phi: POD strain modes (rows: 16 x (tors, bend_y, bend_z))")
    _heat(ax[0, 1], d["M_r"], "M_r = Phi^T J^T m J Phi  [base | modal]")
    _heat(ax[0, 2], d["K_int_r"], "K_int_r = Phi^T K_int Phi (Hooke, modal)")
    _heat(ax[0, 3], d["K_grasp_r"], "K_grasp_r (gripper spring, modal)")
    a = ax[1, 0]
    s = d["singular_values"]
    a.semilogy(np.arange(1, s.size + 1), s, "o-")
    a.axvline(r + 0.5, color="r", ls="--", label=f"r = {r}, energy {float(d['energy_captured']):.4f}")
    a.set_title("POD singular values of the strain snapshots", fontsize=9); a.set_xlabel("mode"); a.legend(fontsize=7)
    a = ax[1, 1]
    idx = np.arange(NB + r)
    a.bar(idx - 0.2, d["M_r_qdd"], 0.4, label="M_r a''")
    a.bar(idx + 0.2, d["f_int_r"], 0.4, label="f_int_r")
    a.axvline(NB - 0.5, color="k", lw=0.5); a.set_title("reduced inertia and internal force", fontsize=9)
    a.set_xlabel("dof (0..5 base, then a)"); a.legend(fontsize=7)
    a = ax[1, 2]
    a.bar(idx - 0.2, d["Jt_lambda_r"], 0.4, label="J_r^T lambda")
    a.bar(idx + 0.2, d["C_r_qdot"], 0.4, label="C_r a'")
    a.axvline(NB - 0.5, color="k", lw=0.5); a.set_title("reduced gripper wrench and damping", fontsize=9)
    a.set_xlabel("dof"); a.legend(fontsize=7)
    a = ax[1, 3]
    sec = np.arange(d["kappa"].size // 3)
    a.plot(sec, d["kappa"][2::3], "o-", label="kappa_z (SOFA strain MO)")
    a.plot(sec, d["kappa_from_modes"][2::3], "x--", label="kappa_0 + Phi a")
    a.set_title("strain reconstructed from the modal coordinates", fontsize=9); a.set_xlabel("section")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="/ros2_ws/artifacts/cable_dynamics")
    parser.add_argument("--drag-steps", type=int, default=100)
    parser.add_argument("--tip-offset", type=float, nargs=2, default=(0.04, 0.06),
                        metavar=("DX_BACK", "DY"), help="tip pulled back along -x and sideways +y [m]")
    parser.add_argument("--rom", action="store_true", help="also dump the POD-Galerkin reduced model")
    parser.add_argument("--modes", default="/ros2_ws/artifacts/cable_mor/cable_strain_modes.txt")
    parser.add_argument("--modes-metadata", default="/ros2_ws/artifacts/cable_mor/cable_strain_modes.yaml")
    parser.add_argument("--n-modes", type=int, default=None, help="default: all modes of the file")
    args = parser.parse_args(argv)
    from cable_identification import cosserat_model as cm
    cfg = cm.load_config(args.config or default_config())
    out = Path(args.output)
    d = collect(cfg, args.drag_steps, args.tip_offset, out / "sofa_export")
    check(d)
    np.savez(out / "cable_dynamics.npz", **d)
    write_report(d, out / "cable_dynamics_report.txt")
    write_figure(d, out / "cable_dynamics.png")
    print((out / "cable_dynamics_report.txt").read_text().split("=== q_base ===")[0])
    print(f"wrote {out}/cable_dynamics.{{npz,png}}, cable_dynamics_report.txt, sofa_export/")
    if args.rom:
        import yaml
        from cable_identification.strain_basis import ReductionSpec
        with open(args.modes_metadata) as stream:
            n_modes = args.n_modes or int(yaml.safe_load(stream)["n_modes"])
        reduction = ReductionSpec(args.modes, args.modes_metadata, n_modes)
        rom = collect_rom(cfg, reduction, args.drag_steps, args.tip_offset, out / "sofa_export_rom")
        check_rom(rom)
        np.savez(out / "cable_dynamics_rom.npz", **rom)
        write_rom_report(rom, out / "cable_dynamics_rom_report.txt")
        write_rom_figure(rom, out / "cable_dynamics_rom.png")
        print((out / "cable_dynamics_rom_report.txt").read_text().split("=== Phi ===")[0])
        print(f"wrote {out}/cable_dynamics_rom.{{npz,png}}, cable_dynamics_rom_report.txt, sofa_export_rom/")
    print("CABLE_DYNAMICS_DUMP_PASSED")


if __name__ == "__main__":
    main()

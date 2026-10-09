#!/usr/bin/env python3
"""Quasi-static TS synthesis: settled shape Jacobians -> certified PDC gains.

The pre-registered fallback of docs/cable_ts_status_and_diagnosis.md §7: the
dynamic 8-state common-quadratic certificate is solver-PROVEN infeasible on
both the wide and the narrow parameter box (basic and relaxed, CLARABEL,
plus fixed-K polish). This layer certifies the quasi-static shape dynamics
instead:

    z_{k+1} = z_k + Ts * (U' J(rho, theta)) v_k,   z = U' q

with J = dq_eq/dg the settled central-difference shape Jacobian measured in
SOFA per premise corner and per physical-parameter vertex, and U the two
most reachable shape directions (SVD). PDC over G_ij = I - Ts J~_i K_j is
the same structure as the working TS-IBVS loop with the deformation Jacobian
in place of the interaction matrix.

The certificate lives in z-space. It is shipped in the STANDARD gains format
by padding: K_j = [K^z_j U' | 0 | 0] (2x8) reproduces u = -sum h_j K^z_j
U'(q - q*) through the unchanged runtime PDC law, and P = blkdiag-lift of
P_z monitors exactly the certified V(z). What the certificate does NOT
cover, by construction: the fast modal transients (quasi-static assumption;
keep the velocity cap low) and targets outside the image of the two static
directions (an arbitrary 3-vector q* is not reachable with two planar
inputs; q* = 0 is).

    ros2 run cable_ts_control settle_shape_jacobians \
        --config .../cable_truth.yaml \
        --model /ros2_ws/artifacts/cable_ts_model_narrow_nofeed.yaml \
        --basis /ros2_ws/artifacts/cable_modal_basis_narrow.yaml \
        --parameter-bounds .../cable_parameter_bounds_narrow.yaml \
        --output /ros2_ws/artifacts/cable_qs_gains_narrow.yaml
"""

import argparse
import itertools
import sys

import numpy as np
import yaml

from cable_ts_control.lmi_synthesis import (
    solve_cable_ts_pdc,
    verify_certificate,
    worst_spectral_radius,
)
from cable_ts_control.modal_basis import ModalBasis
from cable_ts_control.ts_model import CableTsModel


def settle_case(cfg, parameters, targets, sofa_dt, settle_steps, ramp_steps,
                settled_tol):
    """Settled modal-relevant marker shapes at a list of gripper targets.

    One SOFA scene per parameter vertex; the grasp target moves through
    ``targets`` sequentially (ramp + settle each), which matches how the
    runtime reaches nearby operating points. Settledness is asserted, not
    assumed: the last half-second of motion must be below ``settled_tol``.
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
    # Dynamic relaxation: equilibria are damping-independent (Rayleigh forces
    # vanish at rest), so heavy damping speeds convergence without moving the
    # settled shape. The J of damping-paired vertices must agree — checked by
    # the caller's printout.
    cable.set_parameter("rayleigh_mass", 5.0)
    cable.set_parameter("rayleigh_stiffness", 0.5)

    base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    coupling = GraspCoupling(cable, attach_mode="proximity")
    coupling.on_fixture(base)
    for _ in range(50):
        Sofa.Simulation.animate(root, sofa_dt)
    tip = cable.tip_pose()
    coupling.update_grasp(tip, 0.0)

    check = max(1, int(0.5 / sofa_dt))
    max_blocks = max(1, (settle_steps * 5) // check)
    shapes = []
    current = np.asarray(tip[:2], dtype=float)
    for target in targets:
        target = np.asarray(target, dtype=float)
        for k in range(ramp_steps):
            blend = (k + 1) / ramp_steps
            point = (1.0 - blend) * current + blend * target
            coupling.update_grasp(list(point) + [tip[2]] + list(tip[3:7]), 1.0)
            Sofa.Simulation.animate(root, sofa_dt)
        # Adaptive: settle until the last half-second of motion is below the
        # tolerance. Slack corners crawl through a nearly flat buckling
        # valley for tens of seconds; a fixed budget either wastes time on
        # easy probes or fails on those corners.
        before = np.asarray(cable.marker_positions(), dtype=float)[:, :2]
        for block in range(max_blocks):
            for _ in range(check):
                Sofa.Simulation.animate(root, sofa_dt)
            markers = np.asarray(cable.marker_positions(), dtype=float)[:, :2]
            drift = float(np.max(np.linalg.norm(markers - before, axis=1)))
            if drift < settled_tol:
                break
            before = markers
        else:
            raise SystemExit(
                f"not settled at target {target.tolist()} "
                f"(params {parameters}): {drift * 1e3:.2f} mm drift per "
                f"{check} steps after {max_blocks * check} settle steps")
        shapes.append((markers - np.asarray(base[:2])).reshape(-1))
        current = target
    return shapes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="cable truth YAML")
    parser.add_argument("--model", required=True,
                        help="identified TS model YAML; provides the premise "
                             "box, Ts and the state layout the gains pad to")
    parser.add_argument("--basis", required=True, help="modal basis YAML")
    parser.add_argument("--parameter-bounds", required=True)
    parser.add_argument("--output", required=True, help="gains YAML to write")
    parser.add_argument("--output-model", default=None,
                        help="model YAML copy with the (restricted) premise "
                             "box actually certified; required when "
                             "--fore/--bear shrink the box, because the "
                             "runtime memberships must blend over the same "
                             "corners the gains were synthesised for")
    parser.add_argument("--fore", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="restrict the foreshortening range; default: "
                             "the identified model's box")
    parser.add_argument("--bear", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="restrict the bearing range [rad]")
    parser.add_argument("--delta", type=float, default=0.005,
                        help="central-difference gripper perturbation [m]")
    parser.add_argument("--settle-steps", type=int, default=3000,
                        help="cap per target; settling is adaptive")
    parser.add_argument("--ramp-steps", type=int, default=250)
    parser.add_argument("--settled-tol-m", type=float, default=2e-4)
    parser.add_argument("--decay", type=float, default=0.999)
    args = parser.parse_args(argv)

    from cable_identification import cosserat_model as cm
    cfg = cm.load_config(args.config)
    with open(args.parameter_bounds, "r") as f:
        bounds = yaml.safe_load(f)["parameter_bounds"]
    basis = ModalBasis.load(args.basis)
    with open(args.model, "r") as f:
        model_raw = yaml.safe_load(f)
    model = CableTsModel.from_dict(model_raw)
    if model.premise_map.kind != "boundary":
        raise SystemExit("quasi-static synthesis expects boundary premises")

    length = float(model.premise_map.cable_length_m)
    ts = float(model.sample_time)
    sofa_dt = float(cfg["timestep_s"])
    (f_lo, f_hi), (b_lo, b_hi) = model.premise_bounds
    if args.fore is not None:
        f_lo, f_hi = args.fore
    if args.bear is not None:
        b_lo, b_hi = args.bear
    restricted = [args.fore, args.bear] != [None, None]
    if restricted and not args.output_model:
        raise SystemExit("a restricted premise box needs --output-model: the "
                         "runtime must blend over the certified corners")

    # Rule i = corner of the (foreshortening, bearing) box, same bit order as
    # rule_memberships: bit 0 of the premise index is the LAST premise.
    corners = []
    for f_bit, b_bit in itertools.product((0, 1), repeat=2):
        fore = (f_lo, f_hi)[f_bit]
        bear = (b_lo, b_hi)[b_bit]
        radius = (1.0 - fore) * length
        corners.append(radius * np.array([np.cos(bear), np.sin(bear)]))

    # Statics are damping-independent (measured: damping-paired vertices give
    # identical J), so only vertices differing in non-damping parameters get
    # their own SOFA scene.
    names = sorted(bounds)
    vertices = [dict(zip(names, values)) for values in
                itertools.product(*((bounds[n]["min"], bounds[n]["max"])
                                    for n in names))]
    static_names = [n for n in names if not n.startswith("rayleigh")]
    scene_keys = sorted({tuple(v[n] for n in static_names) for v in vertices})

    n_modes = basis.n_modes
    delta = float(args.delta)
    probes = [[delta, 0], [-delta, 0], [0, delta], [0, -delta]]

    print(f"premise box: foreshortening [{f_lo:.4f}, {f_hi:.4f}], "
          f"bearing [{b_lo:.4f}, {b_hi:.4f}] rad"
          + (" (RESTRICTED)" if restricted else ""))
    print(f"{len(scene_keys)} static scenes x {len(corners)} corners, "
          f"delta = {delta * 1e3:.1f} mm, branch-consistent probing")

    boundary_ref = (np.zeros(2) if basis.boundary_reference is None
                    else np.asarray(basis.boundary_reference, dtype=float))

    def project(shape, gripper):
        return basis.project(shape, boundary=np.asarray(gripper) - boundary_ref)

    scene_jacobians = {}
    curvature = 0.0
    branch_drift = 0.0
    for key in scene_keys:
        parameters = dict(zip(static_names, key))
        # Every probe returns to the corner first, so all differences are
        # taken on one settled equilibrium branch; the re-settles at the
        # corner double as a bistability detector.
        targets = []
        for corner in corners:
            targets.append(corner)
            for probe in probes:
                targets.append(np.asarray(corner) + probe)
                targets.append(corner)
        shapes = settle_case(cfg, parameters, targets, sofa_dt,
                             args.settle_steps, args.ramp_steps,
                             args.settled_tol_m)
        per_corner = 1 + 2 * len(probes)
        jac = np.zeros((len(corners), n_modes, 2))
        for i, corner in enumerate(corners):
            block = shapes[i * per_corner:(i + 1) * per_corner]
            block_targets = targets[i * per_corner:(i + 1) * per_corner]
            qs = [project(s, g) for s, g in zip(block, block_targets)]
            q0 = qs[0]
            resettles = qs[2::2]
            drift = max(float(np.linalg.norm(q - q0)) for q in resettles)
            branch_drift = max(branch_drift, drift)
            qxp, qxm, qyp, qym = qs[1], qs[3], qs[5], qs[7]
            jac[i] = np.column_stack(
                [(qxp - qxm) / (2 * delta), (qyp - qym) / (2 * delta)])
            second = max(np.linalg.norm(qxp + qxm - 2 * q0),
                         np.linalg.norm(qyp + qym - 2 * q0))
            first = max(np.linalg.norm(qxp - qxm), np.linalg.norm(qym - qyp))
            ratio = second / max(first, 1e-12)
            curvature = max(curvature, ratio)
            print(f"  {parameters} corner{i}: "
                  f"sigma = {np.linalg.svd(jac[i], compute_uv=False)}, "
                  f"curvature/slope = {ratio:.3f}, "
                  f"branch drift = {drift * 1e3:.3f} mm")
        scene_jacobians[key] = jac

    print(f"worst curvature/slope: {curvature:.3f} "
          f"(<~0.3 for a trustworthy linearisation)")
    print(f"worst branch drift: {branch_drift * 1e3:.3f} mm "
          f"(re-settle reproducibility at the corners)")
    if curvature > 0.5:
        print("ERROR: the settled map is not locally linear at delta; the "
              "quasi-static model is invalid on this box. Shrink the box.")
        return 1

    jacobians = np.stack([
        scene_jacobians[tuple(v[n] for n in static_names)]
        for v in vertices])

    # One common output map: the two most reachable static directions of the
    # mean Jacobian. Per-case U would change the controlled variable with the
    # operating point and void the common certificate.
    u_map = np.linalg.svd(jacobians.reshape(-1, n_modes, 2).mean(axis=0))[0][:, :2]
    b_sets = []
    for v in range(len(vertices)):
        b_list = [ts * (u_map.T @ jacobians[v, i]) for i in range(len(corners))]
        b_sets.append(([np.eye(2)] * len(corners), b_list))
        sigmas = [np.linalg.svd(b, compute_uv=False) for b in b_list]
        print(f"  vtx{v}: sigma(U'J) in "
              f"[{min(s[1] for s in sigmas) / ts:.3f}, "
              f"{max(s[0] for s in sigmas) / ts:.3f}]")

    for mode_relaxed in (False, True):
        cert = solve_cable_ts_pdc(b_sets, relaxed=mode_relaxed,
                                  decay=args.decay)
        kind = "relaxed" if mode_relaxed else "basic"
        print(f"{kind}: {'feasible' if cert.feasible else 'infeasible'} "
              f"({cert.solver_name}, {cert.solver_status})")
        if cert.feasible:
            report = verify_certificate(b_sets, cert)
            radius = worst_spectral_radius(
                b_sets, cert.gains, diagonal_only=mode_relaxed)
            print(f"  verification: "
                  f"{'CERTIFIED' if report['satisfied'] else 'REJECTED'} "
                  f"diag={report['worst_diagonal_residual']:.3e} "
                  f"cross={report['worst_cross_residual']:.3e} "
                  f"rho={radius:.6f}")
            if report["satisfied"] and radius < 1.0:
                break
    else:
        print("ERROR: quasi-static LMIs did not certify; nothing written")
        return 1
    if not (cert.feasible and report["satisfied"] and radius < 1.0):
        print("ERROR: quasi-static LMIs did not certify; nothing written")
        return 1

    # Pad into the runtime 8-state layout: u = -K^z U' (q - q*).
    n = model.state_dim
    gains_padded = []
    for k_z in cert.gains:
        k_full = np.zeros((2, n))
        k_full[:, :n_modes] = k_z @ u_map.T
        gains_padded.append(k_full)
    p_full = np.zeros((n, n))
    p_full[:n_modes, :n_modes] = u_map @ cert.lyapunov @ u_map.T

    payload = {"cable_ts_model": {
        "K": [k.flatten(order="C").tolist() for k in gains_padded],
        "P": p_full.flatten(order="C").tolist(),
        "certificate": f"quasi-static-{cert.certificate_type}",
        "beta": float(cert.beta),
        "solver": cert.solver_name,
        "solver_status": cert.solver_status,
        "verification": {
            "worst_diagonal_residual": float(report["worst_diagonal_residual"]),
            "worst_cross_residual": float(report["worst_cross_residual"]),
            "p_min_eigenvalue": float(report["p_min_eigenvalue"]),
            "worst_spectral_radius": float(radius),
            "spectral_radius_scope":
                "diagonal terms" if cert.certificate_type == "relaxed"
                else "diagonal + averaged cross",
        },
        "quasi_static": {
            "certified_space": "z = U' q, dim 2",
            "output_map_u": u_map.flatten(order="C").tolist(),
            "gains_z": [k.flatten(order="C").tolist() for k in cert.gains],
            "lyapunov_z": cert.lyapunov.flatten(order="C").tolist(),
            "jacobians": jacobians.reshape(len(vertices), len(corners), -1)
                                  .tolist(),
            "delta_m": delta,
            "curvature_over_slope": float(curvature),
            "branch_drift_m": float(branch_drift),
            "premise_bounds": [[f_lo, f_hi], [b_lo, b_hi]],
            "not_covered": "fast modal transients; q* outside Im(U)",
        },
        "sample_time": ts,
    }}
    if cert.slack is not None:
        payload["cable_ts_model"]["quasi_static"]["slack_z"] = \
            cert.slack.flatten(order="C").tolist()
        payload["cable_ts_model"]["max_active_rules"] = \
            int(cert.max_active_rules)
    with open(args.output, "w") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, width=200)
    print(f"wrote {args.output}")

    if args.output_model:
        # Runtime memberships must blend over the corners the gains were
        # certified for. The A/B consequents are copied unchanged: the
        # controller's PDC law never evaluates them, and relabelling would
        # fake an identification that did not happen.
        model_raw["cable_ts_model"]["premise_bounds"] = \
            [[float(f_lo), float(f_hi)], [float(b_lo), float(b_hi)]]
        model_raw["cable_ts_model"]["premise_bounds_note"] = \
            "restricted to the quasi-static certified box by " \
            "settle_shape_jacobians; A/B stem from the wider identified box"
        with open(args.output_model, "w") as f:
            yaml.safe_dump(model_raw, f, default_flow_style=False, width=200)
        print(f"wrote {args.output_model}")

    print(f"||K_z||max = {max(np.linalg.norm(k, 2) for k in cert.gains):.3f}, "
          f"certified rho = {radius:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

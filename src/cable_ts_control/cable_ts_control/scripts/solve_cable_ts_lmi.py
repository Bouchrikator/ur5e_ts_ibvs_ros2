#!/usr/bin/env python3
"""Synthesise the cable TS-PDC gains and verify the Lyapunov conditions.

Step 8 of the plan. Solves one set of gains that stabilises every rule AND
every physical-parameter vertex, then checks the discrete Lyapunov inequalities
before writing anything.

    ros2 run cable_ts_control solve_cable_ts_lmi \
        --model cable_ts_model.yaml --output cable_ts_gains.yaml
"""

import argparse
import sys

import numpy as np
import yaml

from cable_ts_control.lmi_synthesis import (
    certificate_block_residuals,
    solve_cable_ts_pdc,
    solve_cable_ts_pdc_sparse,
    verify_certificate,
    worst_spectral_radius,
)
from cable_ts_control.ts_model import CableTsModel


def load_vertex_sets(model, section):
    """All ``(A_i, B_i)`` families the gains must stabilise."""
    stored = section.get("parameter_bounds") or {}
    if not stored.get("A"):
        return [(model.a_vertices, model.b_vertices)]

    n, m = model.state_dim, model.input_dim
    vertex_sets = []
    for a_group, b_group in zip(stored["A"], stored["B"]):
        vertex_sets.append((
            [np.asarray(a, dtype=float).reshape((n, n)) for a in a_group],
            [np.asarray(b, dtype=float).reshape((n, m)) for b in b_group]))
    return vertex_sets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="cable_ts_model.yaml")
    parser.add_argument("--output", required=True, help="gains YAML to write")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--gain-penalty", type=float, default=1e-3)
    parser.add_argument("--mode", choices=("auto", "basic", "relaxed"),
                        default="auto",
                        help="'basic' is the common quadratic certificate; "
                             "'relaxed' adds the Tanaka-Wang slack Y; 'auto' "
                             "tries basic first and falls back to relaxed")
    parser.add_argument("--max-active-rules", type=int, default=None,
                        help="s in the relaxed conditions; defaults to the "
                             "rule count, since complementary triangular "
                             "memberships leave every rule active inside the box")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--backend", choices=("cvxpy", "sparse"), default="cvxpy",
                        help="'sparse' assembles the same LMIs directly (SCS); "
                             "needed above ~20 states where cvxpy's canonicalisation "
                             "and Clarabel's dense PSD Hessians exhaust the 7 GB host")
    parser.add_argument("--max-iters", type=int, default=100000,
                        help="iteration budget of the sparse backend; a primal residual "
                             "that plateaus long before it is the infeasibility signature")
    parser.add_argument("--report", default=None,
                        help="YAML written on EVERY outcome (attempt statuses, worst "
                             "blocks); the gains file is only written when certified")
    args = parser.parse_args(argv)

    with open(args.model, "r") as f:
        raw = yaml.safe_load(f)
    model = CableTsModel.from_dict(raw)
    vertex_sets = load_vertex_sets(model, raw["cable_ts_model"])

    print(f"synthesising {model.n_rules} PDC gains for {len(vertex_sets)} "
          f"parameter vertices (state {model.state_dim}, input {model.input_dim})")

    attempts = {"auto": ["basic", "relaxed"],
                "basic": ["basic"],
                "relaxed": ["relaxed"]}[args.mode]

    certificate = None
    solve = solve_cable_ts_pdc_sparse if args.backend == "sparse" else solve_cable_ts_pdc
    extra = {"max_iters": args.max_iters} if args.backend == "sparse" else {}
    outcome = {"backend": args.backend, "attempts": [], "certified": False,
               "state_dim": int(model.state_dim), "n_rules": int(model.n_rules),
               "n_parameter_vertices": len(vertex_sets)}

    def write_report():
        if args.report:
            with open(args.report, "w") as f:
                yaml.safe_dump(outcome, f, default_flow_style=False, width=200)

    for attempt in attempts:
        candidate = solve(
            vertex_sets, eps=args.eps, gain_penalty=args.gain_penalty,
            verbose=args.verbose, relaxed=(attempt == "relaxed"),
            max_active_rules=args.max_active_rules, **extra)
        print(f"  {attempt:8s}: "
              f"{'feasible' if candidate.feasible else 'infeasible'}"
              f" (solver={candidate.solver_name}, "
              f"status={candidate.solver_status})")
        record = {"conditions": attempt, "solver": candidate.solver_name,
                  "solver_status": candidate.solver_status, "solver_feasible": bool(candidate.feasible)}
        if candidate.feasible:
            # A first-order "solved" is a claim, not a certificate: verify before accepting.
            check = verify_certificate(vertex_sets, candidate)
            worst = certificate_block_residuals(vertex_sets, candidate)[:3]
            record.update(verified=bool(check["satisfied"]),
                          worst_diagonal_residual=float(check["worst_diagonal_residual"]),
                          worst_cross_residual=float(check["worst_cross_residual"]),
                          worst_blocks=worst)
            print(f"           verification {'OK' if check['satisfied'] else 'FAILED'}: "
                  f"worst block vertex {worst[0]['vertex']} rules "
                  f"({worst[0]['rule_i']},{worst[0]['rule_j']}) residual {worst[0]['residual']:.3e}")
        outcome["attempts"].append(record)
        if candidate.feasible and record["verified"]:
            certificate = candidate
            break

    if certificate is None:
        write_report()
        print("ERROR: no certified PDC gains (solver infeasible or its solution fails the "
              "Lyapunov verification). Widen the sampling period, narrow the parameter "
              "bounds, or re-identify the vertices.")
        return 1

    # Verify EXACTLY the inequalities that were solved. The individual
    # spectral-radius gate on the averaged cross terms is only valid for the
    # basic certificate; under the relaxed theorem those terms may exceed one
    # and still be covered by the slack Q, so there it is restricted to the
    # diagonal terms (which the relaxed conditions do make contractive).
    report = verify_certificate(vertex_sets, certificate)
    relaxed_used = certificate.certificate_type == "relaxed"
    radius = worst_spectral_radius(
        vertex_sets, certificate.gains, diagonal_only=relaxed_used)

    print(f"\ndiscrete Lyapunov verification "
          f"({certificate.certificate_type} conditions, "
          f"beta={certificate.beta}, s={certificate.max_active_rules})")
    print(f"  worst diagonal residual = "
          f"{report['worst_diagonal_residual']:.3e} "
          f"{'< 0 OK' if report['worst_diagonal_residual'] < 0 else '>= 0 FAILED'}")
    print(f"  worst cross residual    = "
          f"{report['worst_cross_residual']:.3e}")
    print(f"  worst residual / ||P||  = "
          f"{report['worst_normalized_residual']:.3e}")
    print(f"  min eig(P) = {report['p_min_eigenvalue']:.3e}"
          + (f", min eig(Q) = {report['q_min_eigenvalue']:.3e}"
             if relaxed_used else ""))
    scope = "diagonal terms" if relaxed_used else "diagonal + averaged cross"
    print(f"  worst |eig| over {scope} = {radius:.6f} "
          f"{'< 1 OK' if radius < 1.0 else '>= 1 FAILED'}")

    if not report["satisfied"] or radius >= 1.0:
        write_report()
        print("\nERROR: the returned gains do not certify stability; "
              "nothing was written.")
        return 1

    outcome["certified"] = True
    write_report()

    for index, gain in enumerate(certificate.gains):
        print(f"  K{index} norm = {np.linalg.norm(gain):.4f}")

    payload = {"cable_ts_model": {
        "K": [g.flatten(order="C").tolist() for g in certificate.gains],
        "P": certificate.lyapunov.flatten(order="C").tolist(),
        "certificate": certificate.certificate_type,
        "beta": float(certificate.beta),
        "solver": certificate.solver_name,
        "solver_status": certificate.solver_status,
        "verification": {
            "worst_diagonal_residual": float(report["worst_diagonal_residual"]),
            "worst_cross_residual": float(report["worst_cross_residual"]),
            "worst_diagonal_residual_at_beta":
                float(report["worst_diagonal_residual_at_beta"]),
            "worst_cross_residual_at_beta":
                float(report["worst_cross_residual_at_beta"]),
            "worst_normalized_residual":
                float(report["worst_normalized_residual"]),
            "p_min_eigenvalue": float(report["p_min_eigenvalue"]),
            "worst_spectral_radius": float(radius),
            "spectral_radius_scope": scope,
        },
        "sample_time": model.sample_time,
    }}
    if relaxed_used:
        payload["cable_ts_model"]["Q"] = \
            certificate.slack.flatten(order="C").tolist()
        payload["cable_ts_model"]["max_active_rules"] = \
            int(certificate.max_active_rules)
        payload["cable_ts_model"]["verification"]["q_min_eigenvalue"] = \
            float(report["q_min_eigenvalue"])
    with open(args.output, "w") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, width=200)

    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

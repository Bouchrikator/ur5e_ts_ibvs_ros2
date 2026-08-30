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
    solve_cable_ts_pdc,
    verify_lyapunov_decrease,
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

    gains = lyapunov = None
    used = None
    for attempt in attempts:
        gains, lyapunov, feasible = solve_cable_ts_pdc(
            vertex_sets, eps=args.eps, gain_penalty=args.gain_penalty,
            verbose=args.verbose, relaxed=(attempt == "relaxed"),
            max_active_rules=args.max_active_rules)
        print(f"  {attempt:8s}: {'feasible' if feasible else 'infeasible'}")
        if feasible:
            used = attempt
            break

    if used is None:
        print("ERROR: the LMIs are infeasible. Widen the sampling period, "
              "narrow the parameter bounds, or re-identify the vertices.")
        return 1

    worst_decrease = verify_lyapunov_decrease(vertex_sets, gains, lyapunov)
    radius = worst_spectral_radius(vertex_sets, gains)

    print("\ndiscrete Lyapunov verification")
    print(f"  max eig(G' P G - P) = {worst_decrease:.3e} "
          f"{'< 0 OK' if worst_decrease < 0 else '>= 0 FAILED'}")
    print(f"  worst |eig(A_i - B_i K_j)| = {radius:.6f} "
          f"{'< 1 OK' if radius < 1.0 else '>= 1 FAILED'}")

    if worst_decrease >= 0.0 or radius >= 1.0:
        print("\nERROR: the returned gains do not certify stability; "
              "nothing was written.")
        return 1

    for index, gain in enumerate(gains):
        print(f"  K{index} norm = {np.linalg.norm(gain):.4f}")

    payload = {"cable_ts_model": {
        "K": [g.flatten(order="C").tolist() for g in gains],
        "P": lyapunov.flatten(order="C").tolist(),
        "certificate": used,
        "max_lyapunov_eigenvalue": float(worst_decrease),
        "worst_spectral_radius": float(radius),
        "sample_time": model.sample_time,
    }}
    with open(args.output, "w") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, width=200)

    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
